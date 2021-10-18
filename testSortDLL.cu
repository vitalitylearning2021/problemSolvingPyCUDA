// nvcc -Xcompiler -fPIC -shared -o testSortDLL.dll testSortDLL.cu

#include <cuda.h>

#include <thrust/sort.h>
#include <thrust/device_ptr.h>

/******************/
/* ERROR CHECKING */
/******************/
#define cudaCHECK(ans) { checkAssert((ans), __FILE__, __LINE__); }
inline void checkAssert(cudaError_t errorCode, const char *file, int line, bool abort = true) {
	if (errorCode != cudaSuccess) {
		fprintf(stderr, "Check assert: %s %s %d\n", cudaGetErrorString(errorCode), file, line);
		if (abort) exit(errorCode);
	}
}

#define DELLEXPORT extern "C" __declspec(dllexport)

/*********************/
/* CUDASORT FUNCTION */
/*********************/
DELLEXPORT void cudaSort(int *h_in, int *h_out, const int N){
	
	int *d_in; cudaCHECK(cudaMalloc(&d_in,  N * sizeof(int)));

	cudaCHECK(cudaMemcpy(d_in, h_in, N * sizeof(int), cudaMemcpyHostToDevice));
	
	thrust::device_ptr<int> d_in_dev_ptr = thrust::device_pointer_cast(d_in);	

	thrust::sort(d_in_dev_ptr, d_in_dev_ptr + N);

	cudaCHECK(cudaMemcpy(h_out, d_in, N * sizeof(int), cudaMemcpyDeviceToHost));

	cudaCHECK(cudaFree(d_in));
}
