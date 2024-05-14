#include <stdio.h>
#include <iostream>
#include <vector>
#include <memory>
#include <stdint.h>

template <class T> 
void restore(T* reciever, T* sender, int Size){
    cudaMemcpy(reciever, sender, Size * sizeof(T), cudaMemcpyDeviceToHost);
    cudaThreadSynchronize();
}

template <class T>
void send(T* reciever, T* sender, int Size){
    cudaMemcpy(reciever, sender, Size * sizeof(T), cudaMemcpyHostToDevice);
    cudaThreadSynchronize();
}

__global__ void kernelSum(const uint8_t* deviceInput, uint32_t* deviceOutput, uint32_t H, uint32_t W, uint32_t BatchLength){
    uint32_t sx = blockDim.x * blockIdx.x + threadIdx.x;
    uint32_t tx = threadIdx.x;
    uint32_t stride;
    printf("%u %u\n", sx, tx);
    __shared__ uint32_t sMemory[64];

    sMemory[tx] = deviceInput[sx];

    __syncthreads();
    for(stride = blockDim.x / 2; stride >= 1; stride >>=1){
        if (tx < stride) {
            sMemory[tx] = sMemory[tx] + sMemory[tx + stride];
        }
        __syncthreads();
    }
    
    if (tx==0U) { 
        deviceOutput[blockIdx.x] = sMemory[tx];
        printf("sum : %d\n", deviceOutput[blockIdx.x]);
    }
}

std::vector<uint8_t> GenerateArry(uint64_t H, uint64_t W, uint64_t BatchLength){
    uint64_t size = H * W * BatchLength;
    std::vector<uint8_t> arr(size, 1);
    return arr;
};

void BatchSum(std::vector<uint8_t> Arr1d, uint64_t H, uint64_t W, uint64_t BatchLength){
    // init
    uint8_t* deviceInput;
    uint32_t* deviceOutput;
    uint32_t* results;
    dim3 block(64, 1, 1);
    dim3 grid(1, 1, 1);

    results = (uint32_t*)malloc(BatchLength * sizeof(uint32_t));
    cudaMalloc((void**)&deviceInput, H * W * BatchLength * sizeof(uint8_t));
    cudaMalloc((void**)&deviceOutput, H * W * BatchLength * sizeof(uint32_t));
    cudaThreadSynchronize();
    
    send(deviceInput, Arr1d.data(), H * W * BatchLength);
    kernelSum<<<grid, block>>>(deviceInput, deviceOutput, H, W, BatchLength);
    restore(results, deviceOutput, H * W * BatchLength);

    free(results);
    cudaFree(deviceInput);
    cudaFree(deviceOutput);
}

int main(){
    uint64_t H = 8;
    uint64_t W = 8;
    uint64_t BatchLength = static_cast<uint64_t>(1);
    //printf("%llu %llu %llu %llu\n", H, W, BatchLength, H * W * BatchLength);
    std::vector<uint8_t> Arr1d = GenerateArry(H, W, BatchLength);
    BatchSum(Arr1d, H, W, BatchLength);
    printf("End\n");
};