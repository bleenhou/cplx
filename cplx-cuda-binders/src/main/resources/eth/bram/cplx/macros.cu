#define GLOBAL_SIZE (blockDim.x * gridDim.x)
#define GLOBAL_IDX (blockIdx.x * blockDim.x + threadIdx.x)
#define BLOCK_IDX (threadIdx.x)
#define BLOCK_SIZE (blockDim.x)