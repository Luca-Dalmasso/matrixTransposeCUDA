/**
 * @file matrixTranspose.cu
 * @brief main application, performs matrix trnspose with different optimizations
 */
 
#include <stdio.h>
#include <stdlib.h>
#include "./inc/common.h"

/**
* @defgroup set of macros for this application (default all to 0)
* @{
*/

/*enable information printing on your GPU*/
#define DEVINFO 1

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

/** @} */


int main(int argc, char **argv){
	#if (DEVINFO)
		deviceInfor();
	#endif
	
	return 0;
}

