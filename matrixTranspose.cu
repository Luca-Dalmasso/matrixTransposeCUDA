/**
 * @file matrixTranspose.cu
 * @brief main application, performs matrix trnspose with different optimizations
 */
 
#include <stdio.h>
#include <stdlib.h>
#include "./inc/common.h"
#include <string.h>

/**
* @defgroup set of macros for this application (default all to 0)
* @{
*/

/*enable verbose stdout (disable this when profiling)*/
#define VERBOSE 0
/*size X of shared memory tile*/
#define BDIMX 16
/*size Y of shared memory tile*/
#define BDIMY 16
/* shared memory padding size (1= used for 4byte banks, 2=used when shared memory has 8byte banks)*/
#define IPAD 1
/*enable host computations for error checking*/
#define CHECK 1

/** @} */

//transpose kernels

/**
* @defgroup matrix transpose kernels
* @{
*/

/**
 * @brief NAIVE row based version of matrix transpose algorithm
 * @param in: source matrix
 * @param out: destination matrix
 * @param nx: #columns
 * @param ny: #rows
 */
__global__ void transposeNaiveRow(float *in, float *out, unsigned int nx, unsigned int ny){
	unsigned int ix=blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int iy=blockDim.y * blockIdx.y + threadIdx.y;
	if (ix>=nx || iy>=ny) return;
	out[ix*ny + iy]=in[iy*nx + ix];
}

/**
 * @brief NAIVE columns based version of matrix transpose algorithm
 * @param in: source matrix
 * @param out: destination matrix
 * @param nx: #columns
 * @param ny: #rows
 */
__global__ void transposeNaiveCol(float *in, float *out, unsigned int nx, unsigned int ny){
	unsigned int ix=blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int iy=blockDim.y * blockIdx.y + threadIdx.y;
	if (ix>=nx || iy>=ny) return;
	out[iy*nx + ix]=in[ix*ny + iy];
}

/**
 * @brief read in rows and write in columns + unroll 4 blocks
 * @param in: source matrix
 * @param out: destination matrix
 * @param nx: #columns
 * @param ny: #rows
 */
__global__ void transposeUnroll4Row(float *in, float *out, unsigned int nx, unsigned int ny)
{
    unsigned int ix = blockDim.x * blockIdx.x * 4 + threadIdx.x;
    unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;

    unsigned int ti = iy * nx + ix; // access in rows
    unsigned int to = ix * ny + iy; // access in columns

    if (ix + 3 * blockDim.x < nx && iy < ny)
    {
        out[to]                   = in[ti];
        out[to + ny * blockDim.x]   = in[ti + blockDim.x];
        out[to + ny * 2 * blockDim.x] = in[ti + 2 * blockDim.x];
        out[to + ny * 3 * blockDim.x] = in[ti + 3 * blockDim.x];
    }
}

/**
 * @brief read in columns and write in rows + unroll 4 blocks
 * @param in: source matrix
 * @param out: destination matrix
 * @param nx: #columns
 * @param ny: #rows
 */
__global__ void transposeUnroll4Col(float *in, float *out, unsigned int nx, unsigned int ny)
{
    unsigned int ix = blockDim.x * blockIdx.x * 4 + threadIdx.x;
    unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;

    unsigned int ti = iy * nx + ix; // access in rows
    unsigned int to = ix * ny + iy; // access in columns

    if (ix + 3 * blockDim.x < nx && iy < ny)
    {
        out[ti]                = in[to];
        out[ti +   blockDim.x] = in[to +   blockDim.x * ny];
        out[ti + 2 * blockDim.x] = in[to + 2 * blockDim.x * ny];
        out[ti + 3 * blockDim.x] = in[to + 3 * blockDim.x * ny];
    }
}

/**
 * @brief read in rows and write in colunms + diagonal coordinate transform,
 * diagonal coordinate system allow to reduce the partition camping phenomenom.
 * @param in: source matrix
 * @param out: destination matrix
 * @param nx: #columns
 * @param ny: #rows
 * @see REAMDE.md: there's a section in which diagonal system and related partition camping issue are explained 
 */
__global__ void transposeDiagonalRow(float *in, float *out, unsigned int nx, unsigned int ny)
{
    unsigned int blk_y = blockIdx.x;
    unsigned int blk_x = (blockIdx.x + blockIdx.y) % gridDim.x;

    unsigned int ix = blockDim.x * blk_x + threadIdx.x;
    unsigned int iy = blockDim.y * blk_y + threadIdx.y;

    if (ix < nx && iy < ny)
    {
        out[ix * ny + iy] = in[iy * nx + ix];
    }
}

/**
 * @brief read in colunms and write in rows + diagonal coordinate transform,
 * diagonal coordinate system allow to reduce the partition camping phenomenom.
 * @param in: source matrix
 * @param out: destination matrix
 * @param nx: #columns
 * @param ny: #rows
 * @see REAMDE.md: there's a section in which diagonal system and related partition camping issue are explained 
 */
__global__ void transposeDiagonalCol(float *in, float *out, unsigned int nx, unsigned int ny)
{
    unsigned int blk_y = blockIdx.x;
    unsigned int blk_x = (blockIdx.x + blockIdx.y) % gridDim.x;

    unsigned int ix = blockDim.x * blk_x + threadIdx.x;
    unsigned int iy = blockDim.y * blk_y + threadIdx.y;

    if (ix < nx && iy < ny)
    {
        out[iy * nx + ix] = in[ix * ny + iy];
    }
}

/**
 * @brief read in rows + write in rows by using shared memory (both access are aligned and coalesced)
 * @param in: source matrix
 * @param out: destination matrix
 * @param nx: #columns
 * @param ny: #rows 
 */
__global__ void transposeSmem(float *in, float *out, unsigned int nx, unsigned int ny)
{
	// static shared memory
    __shared__ float tile[BDIMY][BDIMX];

    // coordinate in original matrix
    unsigned int ix, iy, ti, to;
    ix = blockDim.x * blockIdx.x + threadIdx.x;
    iy = blockDim.y * blockIdx.y + threadIdx.y;

    // linear global memory index for original matrix
    ti = iy * nx + ix;

    // thread index in transposed block
    unsigned int bidx, irow, icol;
    bidx = threadIdx.y * blockDim.x + threadIdx.x;
    irow = bidx / blockDim.y;
    icol = bidx % blockDim.y;

    // coordinate in transposed matrix
    ix = blockDim.y * blockIdx.y + icol;
    iy = blockDim.x * blockIdx.x + irow;

    // linear global memory index for transposed matrix
    to = iy * ny + ix;

    // transpose with boundary test
    if (ix < nx && iy < ny)
    {
        // load data from global memory to shared memory
        tile[threadIdx.y][threadIdx.x] = in[ti];

        // thread synchronization
        __syncthreads();

        // store data to global memory from shared memory
        out[to] = tile[icol][irow];
    }
}

/**
 * @brief shared memory is used with padding to avoid bank conflicts
 * @param in: source matrix
 * @param out: destination matrix
 * @param nx: #columns
 * @param ny: #rows 
 */
__global__ void transposeSmemPad(float *in, float *out, unsigned int nx, unsigned int ny)
{
    // static shared memory with padding
    __shared__ float tile[BDIMY][BDIMX + IPAD];

    // coordinate in original matrix
    unsigned int  ix, iy, ti, to;
    ix = blockDim.x * blockIdx.x + threadIdx.x;
    iy = blockDim.y * blockIdx.y + threadIdx.y;

    // linear global memory index for original matrix
    ti = iy * nx + ix;

    // thread index in transposed block
    unsigned int bidx, irow, icol;
    bidx = threadIdx.y * blockDim.x + threadIdx.x;
    irow = bidx / blockDim.y;
    icol = bidx % blockDim.y;

    // coordinate in transposed matrix
    ix = blockDim.y * blockIdx.y + icol;
    iy = blockDim.x * blockIdx.x + irow;

    // linear global memory index for transposed matrix
    to = iy * ny + ix;

    // transpose with boundary test
    if (ix < nx && iy < ny)
    {
        // load data from global memory to shared memory
        tile[threadIdx.y][threadIdx.x] = in[ti];

        // thread synchronization
        __syncthreads();

        // store data to global memory from shared memory
        out[to] = tile[icol][irow];
    }
}

/**
 * @brief dynamic shared memory used with padding and block unrolling to increase throughput
 * @param in: source matrix
 * @param out: destination matrix
 * @param nx: #columns
 * @param ny: #rows 
 */
__global__ void transposeSmemUnrollPadDyn (float *in, float *out, unsigned int nx, unsigned int ny)
{
    // dynamic shared memory
    extern __shared__ float tile[];

    unsigned int ix = blockDim.x * blockIdx.x * 2 + threadIdx.x;
    unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;

    unsigned int ti = iy * nx + ix;

    unsigned int bidx = threadIdx.y * blockDim.x + threadIdx.x;
    unsigned int irow = bidx / blockDim.y;
    unsigned int icol = bidx % blockDim.y;

    // coordinate in transposed matrix
    unsigned int ix2 = blockDim.y * blockIdx.y + icol;
    unsigned int iy2 = blockDim.x * 2 * blockIdx.x + irow;
    unsigned int to = iy2 * ny + ix2;

    // transpose with boundary test
    if (ix + blockDim.x < nx && iy < ny)
    {
        // load data from global memory to shared memory
        unsigned int row_idx = threadIdx.y * (blockDim.x * 2 + IPAD)+threadIdx.x;
        tile[row_idx] = in[ti];
        tile[row_idx + BDIMX] = in[ti + BDIMX];

        // thread synchronization
        __syncthreads();

        unsigned int col_idx = icol * (blockDim.x * 2 + IPAD) + irow;
        out[to] = tile[col_idx];
        out[to + ny * BDIMX] = tile[col_idx + BDIMX];
    }
}

/** @} */

static void computeHost(float *hSource, float *hDest, unsigned int nx, unsigned ny){
	unsigned int i,j;
	for(i=0;i<ny;i++)
		for(j=0;j<nx;j++)
			hDest[j*nx + i]=hSource[i*nx + j];
}

int main(int argc, char **argv){
	#if (VERBOSE)
		deviceInfor();
	#endif
	
	int iKernel;
	unsigned int nx=1<<11;
	unsigned int ny=1<<11;
	unsigned int blockx=16;
	unsigned int blocky=16;
	unsigned int i;
	
	if (argc<2){
		fprintf(stderr,"usage: <%s> <iKernel> [optional <blockx>] [optional <blocky>] [optional <nx>] [optional <ny>]\n",argv[0]);
		fprintf(stderr,"iKernel=0: copyRow\n");
	    fprintf(stderr,"iKernel=1: copyCol\n");
	    fprintf(stderr,"ikernel=2: transposeNaiveRow\n");
		fprintf(stderr,"ikernel=3: transposeNaiveCol\n");
		fprintf(stderr,"ikernel=4: transposeUnroll4Row\n");
		fprintf(stderr,"ikernel=5: transposeUnroll4Col\n");
		fprintf(stderr,"ikernel=6: transposeDiagonalRow\n");
		fprintf(stderr,"ikernel=7: transposeSmem\n");
		fprintf(stderr,"ikernel=8: transposeSmemPad\n");
		fprintf(stderr,"ikernel=9: transposeSmemUnrollPadDyn\n");
		exit(1);
	}
	
	iKernel=atoi(argv[1]);
	if (argc>2) blockx=atoi(argv[2]);
	if (argc>3) blocky=atoi(argv[3]);
	if (argc>4) nx=atoi(argv[4]);
	if (argc>5) ny=atoi(argv[5]);
	
	dim3 block (blockx, blocky);
    dim3 grid  ((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);
    double iStart, iElaps;
    double effBW;
    
    //data
    float *hSource, *hDest;
    float *dSource, *dDest;
    float *gpuRes;
    //alloc on host
    hSource=(float *)malloc(nx*ny*sizeof(float));
    CHECK_PTR(hSource);
    hDest=(float *)malloc(nx*ny*sizeof(float));
    CHECK_PTR(hDest);
    gpuRes=(float *)malloc(nx*ny*sizeof(float));
    CHECK_PTR(gpuRes);
    //alloc on device
    CHECK_CUDA(cudaMalloc( (void**)&dSource, nx*ny*sizeof(float)));	
    CHECK_CUDA(cudaMalloc( (void**)&dDest, nx*ny*sizeof(float)));
    
    //init on host
    for(i=0;i<nx*ny;i++)
    	hSource[i]=randomUint8()/(float)1.0f;
    //copy on GPU
    CHECK_CUDA(cudaMemcpy(dSource, hSource, nx*ny*sizeof(float), cudaMemcpyHostToDevice));	
    
    #if (VERBOSE)
    fprintf(stdout,"nx=%d, ny=%d, %lu Bytes, grid(%d,%d), block(%d,%d), #threads=%llu\n",nx,ny,
    			(nx*ny*sizeof(float)),grid.x,grid.y,
    			 block.x,block.y,(long long unsigned int)(block.x*block.y*grid.x*grid.y));
    #endif
    
    void (* kernel) (float *, float *, unsigned int, unsigned int);
    char *kernelName;
    
    switch(iKernel){
    	/*setup copyRow*/
    	case 0:
    		#if (VERBOSE)
    		fprintf(stdout,"copyRow kernel selected\n");
    		#endif
    		kernelName=strdup("copyRow");
    		kernel=&copyRow;
    		break;
    	/*setup copyCol*/
    	case 1:
    		#if (VERBOSE)
    		fprintf(stdout,"copyCol kernel selected\n");
    		#endif
    		kernelName=strdup("copyCol");
    		kernel=&copyCol;
    		break;
    	/*setup transposeNaiveRow*/
    	case 2:
    		#if (VERBOSE)
    		fprintf(stdout,"transposeNaiveRow kernel selected\n");
    		#endif
    		kernelName=strdup("transposeNaiveRow");
    		kernel=&transposeNaiveRow;
    		break;
    	/*setup transposeNaiveCol*/
    	case 3:
    		#if (VERBOSE)
    		fprintf(stdout,"transposeNaiveCol kernel selected\n");
    		#endif
    		kernelName=strdup("transposeNaiveCol");
    		kernel=&transposeNaiveCol;
    		break;
    	/*setup transposeUnroll4Row*/
    	case 4:
    		#if (VERBOSE)
    		fprintf(stdout,"transposeUnroll4Row kernel selected\n");
    		#endif
    		kernelName=strdup("transposeUnroll4Row");
    		kernel=&transposeUnroll4Row;
    		grid.x = (nx + block.x * 4 - 1) / (block.x * 4);
    		break;
    	/*setup transposeUnroll4Col*/
    	case 5:
    		#if (VERBOSE)
    		fprintf(stdout,"transposeUnroll4Col kernel selected\n");
    		#endif
    		kernelName=strdup("transposeUnroll4Col");
    		kernel=&transposeUnroll4Col;
    		grid.x = (nx + block.x * 4 - 1) / (block.x * 4);
    		break;
    	/*setup transposeDiagonalRow*/
    	case 6:
    		#if (VERBOSE)
    		fprintf(stdout,"transposeDiagonalRow kernel selected\n");
    		#endif
    		kernelName=strdup("transposeDiagonalRow");
    		kernel=&transposeDiagonalRow;
    		break;
    	/*setup transposeSmem*/
    	case 7:
    		#if (VERBOSE)
    		fprintf(stdout,"transposeSmem kernel selected\n");
    		#endif
    		kernelName=strdup("transposeSmem");
    		kernel=&transposeSmem;
    		break;
    	/*setup transposeSmemPad*/
    	case 8:
    		#if (VERBOSE)
    		fprintf(stdout,"transposeSmemPad kernel selected\n");
    		#endif
    		kernelName=strdup("transposeSmemPad");
    		kernel=&transposeSmemPad;
    		break;
    	/*setup transposeSmemUnrollPadDyn*/
    	case 9:
    		#if (VERBOSE)
    		fprintf(stdout,"transposeSmemUnrollPadDyn kernel selected\n");
    		#endif
    		kernelName=strdup("transposeSmemUnrollPadDyn");
    		kernel=&transposeSmemUnrollPadDyn;
    		grid.x = (nx + block.x * 2 - 1) / (block.x * 2);
    		break;
    	default:
    		#if (VERBOSE)
    		fprintf(stderr,"error in kernel selection, only values between 0-9 are allowed\n");
    		#endif
    		exit(1);
    		break;
    }
    
    if (iKernel==0 || iKernel==1){
    	//run templatized kernels
    	iStart = cpuSecond();
    	kernel<<<grid,block>>>(dSource,dDest,nx,ny);
    	CHECK_CUDA(cudaGetLastError());
		CHECK_CUDA(cudaDeviceSynchronize());
   	 	iElaps = cpuSecond() - iStart;
    }else if(iKernel==9){
    	//run kernel with dynamic shared memory
    	iStart = cpuSecond();
    	kernel<<<grid,block,(BDIMX + IPAD) * BDIMY * sizeof(float)>>>(dSource,dDest,nx,ny);
    	CHECK_CUDA(cudaGetLastError());
		CHECK_CUDA(cudaDeviceSynchronize());
   	 	iElaps = cpuSecond() - iStart;
    }else{
    	//run normal kernel
    	iStart = cpuSecond();
    	kernel<<<grid,block>>>(dSource,dDest,nx,ny);
    	CHECK_CUDA(cudaGetLastError());
		CHECK_CUDA(cudaDeviceSynchronize());
   	 	iElaps = cpuSecond() - iStart;
    }
    
    //get data back from gpu
    CHECK_CUDA(cudaMemcpy(gpuRes, dDest, nx*ny*sizeof(float), cudaMemcpyDeviceToHost));
    
    #if (CHECK)
    //compute result on host
    computeHost(hSource,hDest,nx,ny);
    // check kernel results
    if (iKernel > 1)
    {
        if(checkRes(hDest,gpuRes,nx,ny)==1){
        	fprintf(stderr,"GPU and CPU result missmatch!\n");
        	exit(1);
        }
    }else{
    	if(checkRes(hSource,gpuRes,nx,ny)==1){
        	fprintf(stderr,"GPU and CPU result missmatch!\n");
        	exit(1);
        }
    }
    #endif

    // calculate effective_bandwidth (GB/s)
    effBW=(2 * nx * ny * sizeof(float)) / ((1e+9f)*iElaps);
    /*printf on stdout used for profiling <kernelName>,<elapsedTime>,<bandwidth>,<grid(x,y)>,<block(x,y)>*/
    fprintf(stdout,"%s,%f,%f,grid(%d.%d),block(%d.%d)\n",kernelName, effBW, iElaps, grid.x, grid.y, block.x, block.y);

    // free host and device memory
    CHECK_CUDA(cudaFree(dSource));
    CHECK_CUDA(cudaFree(dDest));
    free(hSource);
    free(hDest);
    free(gpuRes);

    // reset device
    CHECK_CUDA(cudaDeviceReset());
	
	return 0;
}



