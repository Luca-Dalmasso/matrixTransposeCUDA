/**
 * @file common.cu
 * @brief common functions and benchmarks templates for matrix operation
 * @see ../inc/common.h
 */
 
#include "../inc/common.h"
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

template <typename T>
__global__ void copyRow (T *src, T* dest, unsigned int nx, unsigned int ny){
	unsigned int ix=blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int iy=blockDim.y * blockIdx.y + threadIdx.y;
	if (ix>=nx || iy>=ny) return;
	dest[iy*nx + ix]=src[iy*nx + ix];
}

template <typename T>
__global__ void copyCol (T *src, T* dest, unsigned int nx, unsigned int ny){
	unsigned int ix=blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int iy=blockDim.y * blockIdx.y + threadIdx.y;
	if (ix>=nx || iy>=ny) return;
	dest[ix*ny + iy]=src[ix*ny + iy];
}

double cpuSecond(void) {
	struct timeval tp;
	gettimeofday(&tp,NULL);
	return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}

void getDeviceInfo(void){

}

void deviceInfor(void){
	int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0)
    {
        printf("There are no available device(s) that support CUDA\n");
        exit(1);
    }
    printf("Detected %d CUDA Capable device(s)\n", deviceCount);
    int dev = 0, driverVersion = 0, runtimeVersion = 0;
    CHECK_CUDA(cudaSetDevice(dev));
    cudaDeviceProp deviceProp;
    CHECK_CUDA(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Device %d: \"%s\"\n", dev, deviceProp.name);
    cudaDriverGetVersion(&driverVersion);
    cudaRuntimeGetVersion(&runtimeVersion);
    printf("  CUDA Driver Version / Runtime Version          %d.%d / %d.%d\n",
           driverVersion / 1000, (driverVersion % 100) / 10,
           runtimeVersion / 1000, (runtimeVersion % 100) / 10);
    printf("  CUDA Capability Major/Minor version number:    %d.%d\n",
           deviceProp.major, deviceProp.minor);
    printf("  Total amount of global memory:                 %.2f MBytes (%llu "
           "bytes)\n", (float)deviceProp.totalGlobalMem / pow(1024.0, 3),
           (unsigned long long)deviceProp.totalGlobalMem);
    printf("  GPU Clock rate:                                %.0f MHz (%0.2f "
           "GHz)\n", deviceProp.clockRate * 1e-3f,
           deviceProp.clockRate * 1e-6f);
    printf("  Memory Clock rate:                             %.0f Mhz\n",
           deviceProp.memoryClockRate * 1e-3f);
    printf("  Memory Bus Width:                              %d-bit\n",
           deviceProp.memoryBusWidth);
    if (deviceProp.l2CacheSize)
    {
        printf("  L2 Cache Size:                                 %d bytes\n",
               deviceProp.l2CacheSize);
    }

    printf("  Max Texture Dimension Size (x,y,z)             1D=(%d), "
           "2D=(%d,%d), 3D=(%d,%d,%d)\n", deviceProp.maxTexture1D,
           deviceProp.maxTexture2D[0], deviceProp.maxTexture2D[1],
           deviceProp.maxTexture3D[0], deviceProp.maxTexture3D[1],
           deviceProp.maxTexture3D[2]);
    printf("  Max Layered Texture Size (dim) x layers        1D=(%d) x %d, "
           "2D=(%d,%d) x %d\n", deviceProp.maxTexture1DLayered[0],
           deviceProp.maxTexture1DLayered[1], deviceProp.maxTexture2DLayered[0],
           deviceProp.maxTexture2DLayered[1],
           deviceProp.maxTexture2DLayered[2]);
    printf("  Total amount of constant memory:               %lu bytes\n",
           deviceProp.totalConstMem);
    printf("  Total amount of shared memory per block:       %lu bytes\n",
           deviceProp.sharedMemPerBlock);
    printf("  Total number of registers available per block: %d\n",
           deviceProp.regsPerBlock);
    printf("  Warp size:                                     %d\n",
           deviceProp.warpSize);
    printf("  Maximum number of threads per multiprocessor:  %d\n",
           deviceProp.maxThreadsPerMultiProcessor);
    printf("  Maximum number of threads per block:           %d\n",
           deviceProp.maxThreadsPerBlock);
    printf("  Maximum sizes of each dimension of a block:    %d x %d x %d\n",
           deviceProp.maxThreadsDim[0],
           deviceProp.maxThreadsDim[1],
           deviceProp.maxThreadsDim[2]);
    printf("  Maximum sizes of each dimension of a grid:     %d x %d x %d\n",
           deviceProp.maxGridSize[0],
           deviceProp.maxGridSize[1],
           deviceProp.maxGridSize[2]);
    printf("  Maximum memory pitch:                          %lu bytes\n",
           deviceProp.memPitch);
}

uint_8 randomUint8(void){
	time_t t;
	srand((unsigned) time(&t));
	return rand()%0xff;
}


void testBenchmarks (unsigned int bx, unsigned int by){
	double tFast, tSlow;
	double bandwidth; 
	unsigned int sizeX=(1<<20);
	/*General CUDA Kernel 2D configuration: BLOCK*/
	dim3 block (bx, by);
	/*General CUDA Kernel 2D configuration: GRID*/
	dim3 grid ((sizeX-1+block.x)/block.x,(sizeX-1+block.y)/block.y);
	fprintf(stdout,"Benchmark kernel configurations: Grid(%d,%d), Block(%d,%d), size %d\n",grid.x,grid.y,block.x,block.y,sizeX*sizeX);
	unsigned int i;
	//************************
	//********FLOAT***********
	//************************
	float *hSource;
	float *dSource, *dDest;
	float *gpuRes;
	
	hSource=(float *)malloc(sizeX*sizeof(float));
	CHECK_PTR(hSource);
	gpuRes=(float *)malloc(sizeX*sizeof(float));
	CHECK_PTR(gpuRes);
	CHECK_CUDA(cudaMalloc( (void**)&dSource, sizeX*sizeof(float)));	
	CHECK_CUDA(cudaMalloc( (void**)&dDest, sizeX*sizeof(float)));
	for(i=0;i<sizeX;i++)
		hSource[i]=randomUint8()/(float)(1.0);
	CHECK_CUDA(cudaMemcpy(dSource, hSource, sizeX*sizeof(float), cudaMemcpyHostToDevice));
	tFast=cpuSecond();
	copyRow<float><<<grid,block>>>(dSource, dDest, sizeX, sizeX);
	CHECK_CUDA(cudaGetLastError());
	cudaDeviceSynchronize();
	tFast=cpuSecond()-tFast;
	CHECK_CUDA(cudaMemcpy(gpuRes, dDest, sizeX*sizeof(float), cudaMemcpyDeviceToHost));
	for(i=0;i<sizeX;i++){
		if(hSource!=gpuRes){
			fprintf(stderr,"GPU bad result!\n");
			exit(1);
		}
	}
	//test float strided
	tSlow=cpuSecond();
	copyCol<float><<<grid,block>>>(dSource, dDest, sizeX, sizeX);
	CHECK_CUDA(cudaGetLastError());
	cudaDeviceSynchronize();
	tSlow=cpuSecond()-tSlow;
	//result
	bandwidth=(sizeX*sizeof(float)*2)/(tFast*(1e+9f));
	fprintf(stdout,"Floating point single precision upper bandwidth: %fGB/s\n",bandwidth);
	bandwidth=(sizeX*sizeof(float)*2)/(tSlow*(1e+9f));
	fprintf(stdout,"Floating point single precision lower bandwidth: %fGB/s\n",bandwidth);
	
	free(hSource);
	free(gpuRes);
	CHECK_CUDA(cudaFree(dSource));
	CHECK_CUDA(cudaFree(dDest));
	
	
}




