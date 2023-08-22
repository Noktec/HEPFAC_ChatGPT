#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>

// Define the trie node structure
typedef struct node {
    int bitmap[8];  // Bitmap for 256 possible children
    int offset;     // Offset in the flat trie array to the first child
    int is_final;   // Indicates if this node is the end of a pattern
    struct node* children[256];  // Pointers to child nodes
} node_t;

// Global variables
node_t flat_trie[10000];
int node_count = 0;

node_t* create_node() {
    node_t* new_node = (node_t*)malloc(sizeof(node_t));
    memset(new_node, 0, sizeof(node_t));
    return new_node;
}

void insert_pattern(node_t* root, const char* pattern) {
    node_t* current = root;
    while (*pattern) {
        unsigned char index = *pattern;
        if (!current->children[index]) {
            current->children[index] = create_node();
        }
        current->bitmap[index / 32] |= (1 << (index % 32));
        current = current->children[index];
        pattern++;
    }
    current->is_final = 1;
}

// Function to flatten the trie into an array
int flatten_trie(node_t* root) {
    if (!root) return -1;

    int index = node_count++;
    flat_trie[index] = *root;

    for (int i = 0; i < 256; i++) {
        if (root->children[i]) {
            root->offset = flatten_trie(root->children[i]);
        }
    }
    return index;
}

// CUDA kernel for the optimized search
__global__ void optimized_gpu_search_kernel(node_t* flat_trie, char* text, int text_len, int segment_len, int* results) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int start = tid * segment_len;
    int end = min(start + segment_len, text_len);

    __shared__ char shared_text[1024];  // Assuming blockDim.x = 256 and segment_len = 4
    int local_tid = threadIdx.x;
    for (int i = 0; i < segment_len; i += blockDim.x) {
        if (local_tid + i < segment_len && start + i + local_tid < text_len) {
            shared_text[local_tid + i] = text[start + i + local_tid];
        }
    }
    __syncthreads();

    for (int i = 0; i < segment_len && start + i < text_len; i++) {
        int current_index = 0;  // Start from root
        for (int j = i; j < segment_len && current_index != -1; j++) {
            unsigned char ch = shared_text[j];
            if ((flat_trie[current_index].bitmap[ch / 32] & (1 << (ch % 32))) == 0) {
                break;  // Character not in trie
            }
            current_index = flat_trie[current_index].offset;  // Move to child
            if (flat_trie[current_index].is_final) {
                atomicAdd(&results[start + j], 1);  // Found a pattern ending at j
            }
        }
    }
}

void optimized_gpu_search(node_t* trie, char* text, int text_len) {
    // Flatten the trie
    node_count = 0;
    flatten_trie(trie);

    // Allocate memory on the GPU
    node_t* d_trie;
    char* d_text;
    int* d_results;
    cudaMalloc(&d_trie, node_count * sizeof(node_t));
    cudaMalloc(&d_text, text_len * sizeof(char));
    cudaMalloc(&d_results, text_len * sizeof(int));

    // Copy data to the GPU
    cudaMemcpy(d_trie, flat_trie, node_count * sizeof(node_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_text, text, text_len * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemset(d_results, 0, text_len * sizeof(int));

    // Determine grid and block sizes
    int segment_len = 4;  // Length of text segment for each thread
    int threads_per_block = 256;
    int blocks = (text_len + segment_len - 1) / segment_len;

    // Launch the optimized kernel
    optimized_gpu_search_kernel<<<blocks, threads_per_block>>>(d_trie, d_text, text_len, segment_len, d_results);

    // Copy results back to the host and process them
    int results[text_len];
    cudaMemcpy(results, d_results, text_len * sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < text_len; i++) {
        if (results[i] > 0) {
            printf("Pattern found ending at position %d\n", i);
        }
    }

    // Cleanup
    cudaFree(d_trie);
    cudaFree(d_text);
    cudaFree(d_results);
}

// CPU search function
void cpu_search(node_t* root, char* text, int text_len) {
    for (int i = 0; i < text_len; i++) {
        node_t* current = root;
        for (int j = i; j < text_len && current; j++) {
            unsigned char ch = text[j];
            if ((current->bitmap[ch / 32] & (1 << (ch % 32))) == 0) {
                break;  // Character not in trie
            }
            current = current->children[ch];
            if (current && current->is_final) {
                printf("Pattern found ending at position %d\n", j);
            }
        }
    }
}

void free_trie(node_t* root) {
    if (!root) return;
    for (int i = 0; i < 256; i++) {
        if (root->children[i]) {
            free_trie(root->children[i]);
        }
    }
    free(root);
}


int main(int argc, char* argv[]) {
    if (argc != 4) {
        printf("Usage: %s <text_file> <patterns_file> <search_mode>\n", argv[0]);
        printf("search_mode can be 'cpu' or 'gpu'\n");
        return -1;
    }

    // Load text from argv[1]
    FILE* text_file = fopen(argv[1], "r");
    fseek(text_file, 0, SEEK_END);
    int text_len = ftell(text_file);
    fseek(text_file, 0, SEEK_SET);
    char* text = (char*)malloc(text_len + 1);
    fread(text, 1, text_len, text_file);
    text[text_len] = '\0';  // Null-terminate the string
    fclose(text_file);

    // Build the trie using the patterns from argv[2]
    node_t* root = create_node();
    FILE* patterns_file = fopen(argv[2], "r");
    char pattern[256];  // Assuming max pattern length is 255
    while (fgets(pattern, sizeof(pattern), patterns_file)) {
        pattern[strcspn(pattern, "\n")] = 0;  // Remove newline character
        insert_pattern(root, pattern);
    }
    fclose(patterns_file);

    // Decide the search mode based on argv[3]
    if (strcmp(argv[3], "cpu") == 0) {
        cpu_search(root, text, text_len);
    } else if (strcmp(argv[3], "gpu") == 0) {
        optimized_gpu_search(root, text, text_len);
    } else {
        printf("Invalid search mode! Choose 'cpu' or 'gpu'.\n");
        return -1;
    }

    // Cleanup
    free(text);
    free_trie(root);


    return 0;
}
