#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <windows.h>
using namespace std;


template<class T>
__global__ void transpose(T* device_ouput, const T* device_input, int w, int h){
    const int sx = blockDim.x * blockIdx.x + threadIdx.x;
    const int sy = blockDim.y * blockIdx.y + threadIdx.y;
    if(w<=sx||h<=sy){return;}
    device_ouput[sx*h+sy] = device_input[sy*w+sx];
}

template<class T>
void send(T *reciver, T *sender, int data_size){
    cudaMemcpy(reciver, sender, sizeof(T)*data_size, cudaMemcpyHostToDevice);
    cudaThreadSynchronize();
}

template<class T>
void restore(T *reciver, T *sender, int data_size){
    cudaMemcpy(reciver, sender, sizeof(T)*data_size, cudaMemcpyDeviceToHost);
    cudaThreadSynchronize();
}



int main(){
    int H=4;
    int W=4;
    dim3 block(1, 1, 1);
    dim3 grid(H, W, 1);

    // original_data and result; 
    int *A = (int*)malloc(H*W*sizeof(int));
    int *At = (int*)malloc(H*W*sizeof(int));
    
    int *device_input;
    int *device_output;
    cudaMalloc((void**)&device_input, H*W*sizeof(int));
    cudaMalloc((void**)&device_output, H*W*sizeof(int));
    // init
    for(int i=0;i<H*W;i++){A[i]=i;}
    // task
    send(device_input, A, H*W);
    transpose<<<block,grid>>>(device_output, device_input, W, H);
    restore(At, device_output, H*W);

    for(int i=0;i<H;i++){
        for(int j=0;j<W;j++){
            cout<<At[j+W*i]<<" ";
        }
        cout<<endl;
    }
    free(A);
    free(At);
    cudaFree(device_output);
    cudaFree(device_input);
}