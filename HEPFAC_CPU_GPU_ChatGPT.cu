#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ALPHABET_SIZE 256
#define MAX_DEPTH 8

typedef struct TrieNode {
    struct TrieNode* children[ALPHABET_SIZE];
    int is_end_of_word;
    unsigned int bitmap[8];  // Assuming 32-bit integers and 256 ASCII characters
} TrieNode;

typedef struct {
    int children[ALPHABET_SIZE];  // Indices to children in the flat array
    int is_end_of_word;
    unsigned int bitmap[8];  // Assuming 32-bit integers and 256 ASCII characters
} FlatTrieNode;


TrieNode* create_node() {
    TrieNode* node = (TrieNode*) malloc(sizeof(TrieNode));
    memset(node, 0, sizeof(TrieNode));
    return node;
}

void insert(TrieNode* root, const char* pattern) {
    TrieNode* node = root;
    int depth = 0;
    while (*pattern && depth < MAX_DEPTH) {
        unsigned char index = (unsigned char) *pattern;
        if (!node->children[index]) {
            node->children[index] = create_node();
        }
        node->bitmap[index / 32] |= (1 << (index % 32));
        node = node->children[index];
        pattern++;
        depth++;
    }
    node->is_end_of_word = 1;
}

// Generate a unique signature for each subtree
unsigned long generate_signature(TrieNode* node) {
    if (!node) return 0;
    
    unsigned long signature = 5381;  // Starting number
    signature = ((signature << 5) + signature) + node->character;
    for (int i = 0; i < 256; i++) {
        if (node->children[i]) {
            signature = ((signature << 5) + signature) + generate_signature(node->children[i]);
        }
    }
    return signature;
}

void compress_suffixes(TrieNode* node, TrieNode* root) {
    if (!node) return;

    unsigned long current_signature = generate_signature(node);
    
    // Compare with every other node's subtree
    for (int i = 0; i < 256; i++) {
        if (node->children[i]) {
            unsigned long child_signature = generate_signature(node->children[i]);
            if (child_signature == current_signature) {
                // Free the subtree of node->children[i] and point it to node
                // Note: actual node deletion code is omitted for simplicity
                node->children[i] = node;
            }
        }
    }
    
    // Recurse for all children
    for (int i = 0; i < 256; i++) {
        compress_suffixes(node->children[i], root);
    }
}


// Merge nodes with only one child (prefix compression)
void compress_trie(TrieNode* node) {
    if (!node) {
        return;
    }

    int children_count = 0;
    unsigned char child_index = 0;
    for (unsigned char i = 0; i < ALPHABET_SIZE; i++) {
        if (node->children[i]) {
            children_count++;
            child_index = i;
        }
    }

    // If only one child, merge with current node
    if (children_count == 1) {
        TrieNode* child = node->children[child_index];
        memcpy(node->bitmap, child->bitmap, sizeof(node->bitmap));
        for (unsigned char i = 0; i < ALPHABET_SIZE; i++) {
            node->children[i] = child->children[i];
        }
        free(child);
    }

    // Recursively compress child nodes
    for (unsigned char i = 0; i < ALPHABET_SIZE; i++) {
        compress_trie(node->children[i]);
    }
}


// Recursive function to free trie memory
void free_trie(TrieNode* node) {
    if (!node) {
        return;
    }
    for (unsigned char i = 0; i < ALPHABET_SIZE; i++) {
        free_trie(node->children[i]);
    }
    free(node);
}

void cpu_search(TrieNode* root, const char* text) {
    const char* ptr = text;
    while (*ptr) {
        TrieNode* node = root;
        const char* temp = ptr;
        while (*temp && node->children[(unsigned char)*temp]) {
            unsigned char index = (unsigned char) *temp;
            if (!(node->bitmap[index / 32] & (1 << (index % 32)))) {
                break;  // Character not in trie
            }
            node = node->children[index];
            temp++;
            if (node->is_end_of_word) {
                printf("Pattern found at position %ld\n", ptr - text);
            }
        }
        ptr++;
    }
}

__global__ void search_kernel(FlatTrieNode* d_trie, const char* d_text, int text_len, int* d_results) {
    extern __shared__ char shared_text[];

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int local_tid = threadIdx.x;

    // Load text into shared memory
    if (tid < text_len) {
        shared_text[local_tid] = d_text[tid];
    }
    __syncthreads();

    int position = local_tid;
    while (position < text_len) {
        FlatTrieNode* node = &d_trie[0];  // Start at the root
        int offset = 0;
        while (node && position + offset < text_len) {
            char ch = shared_text[position + offset];
            if (!(node->bitmap[(unsigned char)ch / 32] & (1 << ((unsigned char)ch % 32)))) {
                break;  // Character not in trie
            }
            node = &d_trie[node->children[(unsigned char)ch]];
            offset++;
            if (node && node->is_end_of_word) {
                atomicAdd(&d_results[position], 1);
            }
        }
        position += blockDim.x;
    }
}

void optimized_gpu_search(FlatTrieNode* flat_trie, int trie_size, const char* text) {
    FlatTrieNode* d_trie;
    char* d_text;
    int text_len = strlen(text);
    int* d_results;

    // Allocate memory on GPU
    cudaMalloc(&d_trie, trie_size * sizeof(FlatTrieNode));
    cudaMalloc(&d_text, text_len + 1);
    cudaMalloc(&d_results, text_len * sizeof(int));

    // Copy data to GPU memory
    cudaMemcpy(d_trie, flat_trie, trie_size * sizeof(FlatTrieNode), cudaMemcpyHostToDevice);
    cudaMemcpy(d_text, text, text_len + 1, cudaMemcpyHostToDevice);

    // Launch CUDA kernel
    int numThreadsPerBlock = 256;
    int numBlocks = (text_len + numThreadsPerBlock - 1) / numThreadsPerBlock;
    search_kernel<<<numBlocks, blockSize>>>(d_flat_trie, d_text, d_results);

    // Copy results back to CPU
    int results[text_len];
    cudaMemcpy(results, d_results, text_len * sizeof(int), cudaMemcpyDeviceToHost);

    // Process the results array to identify matches and their positions
    for (int i = 0; i < text_len; i++) {
        if (results[i] > 0) {
            printf("Pattern found at position %d\n", i);
        }
    }

    // Cleanup GPU memory
    cudaFree(d_trie);
    cudaFree(d_text);
    cudaFree(d_results);
}


int main(int argc, char* argv[]) {
    if (argc != 4) {
        printf("Usage: %s <text_file> <patterns_file> <search_mode>\n", argv[0]);
        printf("search_mode can be 'cpu' or 'gpu'\n");
        return -1;
    }

    TrieNode* root = create_node();

    // Load patterns from argv[2] and insert into trie
    FILE* patterns_file = fopen(argv[2], "r");
    char pattern[256];
    while (fgets(pattern, sizeof(pattern), patterns_file)) {
        pattern[strcspn(pattern, "\n")] = 0;  // Remove newline character
        insert(root, pattern);
    }
    fclose(patterns_file);

    compress_trie(root);
    compress_suffixes(root, root);


    // Load text from argv[1]
    FILE* text_file = fopen(argv[1], "r");
    fseek(text_file, 0, SEEK_END);
    int text_len = ftell(text_file);
    fseek(text_file, 0, SEEK_SET);
    char* text = (char*)malloc(text_len + 1);
    fread(text, 1, text_len, text_file);
    text[text_len] = '\0';
    fclose(text_file);

    if (strcmp(argv[3], "cpu") == 0) {
        cpu_search(root, text);
    } else if (strcmp(argv[3], "gpu") == 0) {
        optimized_gpu_search(root, text);
    } else {
        printf("Invalid search mode! Choose 'cpu' or 'gpu'.\n");
        return -1;
    }

    free(text);
    free_trie(root);

    return 0;
}
