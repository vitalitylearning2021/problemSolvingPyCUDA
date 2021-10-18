// nvcc -Xcompiler -fPIC -shared -o testSortByKeyDLL.dll testSortByKeyDLL.cu

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

/**************************/
/* CUDASORTBYKEY FUNCTION */
/**************************/
DELLEXPORT void cudaSortByKey(int *h_key_in, float2 *h_val_in, int *h_key_out, float2 *h_val_out, const int N){
	
	int *d_key_in; 		cudaCHECK(cudaMalloc(&d_key_in,   N * sizeof(int)));
	int *d_key_out; 	cudaCHECK(cudaMalloc(&d_key_out,  N * sizeof(int)));
	float2 *d_val_in; 	cudaCHECK(cudaMalloc(&d_val_in,   N * sizeof(float2)));
	float2 *d_val_out; 	cudaCHECK(cudaMalloc(&d_val_out,  N * sizeof(float2)));

	cudaCHECK(cudaMemcpy(d_key_in, h_key_in, N * sizeof(int),    cudaMemcpyHostToDevice));
	cudaCHECK(cudaMemcpy(d_val_in, h_val_in, N * sizeof(float2), cudaMemcpyHostToDevice));
	
	thrust::device_ptr<int> 	d_key_in_dev_ptr = thrust::device_pointer_cast(d_key_in);	
	thrust::device_ptr<float2> 	d_val_in_dev_ptr = thrust::device_pointer_cast(d_val_in);	

	thrust::sort_by_key(d_key_in_dev_ptr, d_key_in_dev_ptr + N, d_val_in_dev_ptr);

	cudaCHECK(cudaMemcpy(h_key_out, d_key_in, N * sizeof(int),    cudaMemcpyDeviceToHost));
	cudaCHECK(cudaMemcpy(h_val_out, d_val_in, N * sizeof(float2), cudaMemcpyDeviceToHost));

	cudaCHECK(cudaFree(d_key_in));
	cudaCHECK(cudaFree(d_val_in));
	cudaCHECK(cudaFree(d_key_out));
	cudaCHECK(cudaFree(d_val_out));
}