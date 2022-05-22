/**
 * @file common.h
 * @brief library for common used functions and macros in a CUDA applications
 */

#ifndef _COMMON
#define _COMMON

/**
* @defgroup C typedef for this application
* @{
*/
typedef unsigned char uint_8;
typedef signed char int_8;
/** @} */

/**
 * @brief check if the cuda call correctly worked
 * @param error: return value of a systemcall
 */
#define CHECK_CUDA(error)                                                       \
{                                                                              	\
    if (error != cudaSuccess)                                                  	\
    {                                                                          	\
        fprintf(stderr, "Error: %s:%d\n", __FILE__, __LINE__);                 	\
        fprintf(stderr, "code: %d, reason: %s\n", error,                       	\
        cudaGetErrorString(error));                                   			\
		exit(-1);						       									\
    }                                                                          	\
    																			\
}


/**
 * @brief check pointer validity
 * @param ptr: generic pointer
 */
#define CHECK_PTR(ptr)                                                          \
{                                                                              	\
    if (ptr == NULL)                                                  			\
    {                                                                          	\
        fprintf(stderr, "Error: %s:%d\n", __FILE__, __LINE__);                 	\
        fprintf(stderr, "Null pointer\n" );				                        \
		exit(-1);						       									\
    }                                                                          	\
    																			\
}

/**
 * @brief function to compute upper bound peak global memory bandwidth
 * load and store coalesced and sequential access from global memory (nx*sizeof(T) Bytes)
 * @param src: source array
 * @param dest: destination array
 * @param unsigned int nx: array's index x
 * @param unsigned int ny: array's index y
 */
__global__ void copyRow (float *src, float* dest, unsigned int nx, unsigned int ny);

/**
 * @brief function to compute lower bound global memory bandwidth
 * load and store strided non coalesced access from global memory (nx*sizeof(T) Bytes)
 * @param src: source array
 * @param dest :  destination array
 * @param unsigned int nx: array's index x
 * @param unsigned int ny: array's index y
 */
__global__ void copyCol (float *src, float* dest, unsigned int nx, unsigned int ny);

/**
 * @brief function that returns a random number in range [0-255]
 */
uint_8 randomUint8(void);

/**
 * @brief function that returns time in seconds using gettimeofday system call to get system's clock
 * usage: tStart=cpuSecond(); [some computations] tElapsed=cpuSecond()-tStart;
 */
double cpuSecond(void);

/**
 * @Brief function used to check if gpu and cpu results are the same
 * @param host: host result in form of array or matrix
 * @param device: device result in form of array o matrix
 * @param nx: array's size X
 * @param ny: number of rows of the matrix (PUT 1 if 1D array)
 * @return 0 if equals, 1 if NOT
 */
uint_8 checkRes(float *host, float *device, unsigned int nx, unsigned int ny);

/**
 * @brief query info from your GPU
 */
void deviceInfor(void);


#endif















