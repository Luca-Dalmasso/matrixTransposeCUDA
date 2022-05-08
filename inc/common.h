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
 * @brief template function to compute upper bound peak global memory bandwidth
 * load and store coalesced and sequential access from global memory (nx*sizeof(T) Bytes)
 * @param src *T: templatized source array
 * @param dest *T: templatized destination array
 * @param unsigned int nx: array's index x
 * @param unsigned int ny: array's index y
 * @note *T can be of any type
 */
template <typename T>
__global__ void copyRow (T *src, T* dest, unsigned int nx, unsigned int ny);

/**
 * @brief template function to compute lower bound global memory bandwidth
 * load and store strided non coalesced access from global memory (nx*sizeof(T) Bytes)
 * @param src *T: templatized source array
 * @param dest *T: templatized destination array
 * @param unsigned int nx: array's index x
 * @param unsigned int ny: array's index y
 * @note *T can be of any type
 */
template <typename T>
__global__ void copyCol (T *src, T* dest, unsigned int nx, unsigned int ny);


/**
 * @brief wrapper function for benchmarks, shows what are (approximately) the highest and lowest
 * bandwidth for a typical GPU application involving matrixes.
 * @param bx: block size X
 * @param by: block size Y
 */
void testBenchmarks (unsigned int bx, unsigned int by);

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
 * @Brief function that print some information on a CUDA compatible GPU
 */
void deviceInfor(void);


#endif















