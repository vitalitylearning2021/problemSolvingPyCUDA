// nvcc -Xcompiler -fPIC -shared -o testSortByKeyZippedDLL.dll testSortByKeyZippedDLL.cu

#include <cuda.h>

#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/iterator/zip_iterator.h>

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

/***********************/
/* CUDASORTBYKEYZIPPED */
/***********************/
DELLEXPORT void cudaSortByKeyZipped(int *h_key_in, float2 *h_val1_in, float2 *h_val2_in, int *h_key_out, float2 *h_val1_out, float2 *h_val2_out, const int N){
	
	int *d_key_in; 		cudaCHECK(cudaMalloc(&d_key_in,   N * sizeof(int)));
	int *d_key_out; 	cudaCHECK(cudaMalloc(&d_key_out,  N * sizeof(int)));
	float2 *d_val1_in; 	cudaCHECK(cudaMalloc(&d_val1_in,  N * sizeof(float2)));
	float2 *d_val2_in; 	cudaCHECK(cudaMalloc(&d_val2_in,  N * sizeof(float2)));
	float2 *d_val1_out; 	cudaCHECK(cudaMalloc(&d_val1_out, N * sizeof(float2)));
	float2 *d_val2_out; 	cudaCHECK(cudaMalloc(&d_val2_out, N * sizeof(float2)));

	cudaCHECK(cudaMemcpy(d_key_in,  h_key_in,  N * sizeof(int),    cudaMemcpyHostToDevice));
	cudaCHECK(cudaMemcpy(d_val1_in, h_val1_in, N * sizeof(float2), cudaMemcpyHostToDevice));
	cudaCHECK(cudaMemcpy(d_val2_in, h_val2_in, N * sizeof(float2), cudaMemcpyHostToDevice));
	
	thrust::device_ptr<int> 	d_key_in_dev_ptr  = thrust::device_pointer_cast(d_key_in);	
	thrust::device_ptr<float2> 	d_val1_in_dev_ptr = thrust::device_pointer_cast(d_val1_in);	
	thrust::device_ptr<float2> 	d_val2_in_dev_ptr = thrust::device_pointer_cast(d_val2_in);	

    	typedef thrust::device_vector<float2>::iterator         Float2Iterator;
    	typedef thrust::tuple<Float2Iterator, Float2Iterator> 	Float2IteratorTuple;
    	typedef thrust::zip_iterator<Float2IteratorTuple>       ZippedFloat2Iterator;

    	ZippedFloat2Iterator A_first = thrust::make_zip_iterator(thrust::make_tuple(d_val1_in_dev_ptr,     d_val2_in_dev_ptr));

	thrust::sort_by_key(d_key_in_dev_ptr, d_key_in_dev_ptr + N, A_first);

	cudaCHECK(cudaMemcpy(h_key_out, d_key_in, N * sizeof(int),    cudaMemcpyDeviceToHost));
	cudaCHECK(cudaMemcpy(h_val1_out, d_val1_in, N * sizeof(float2), cudaMemcpyDeviceToHost));
	cudaCHECK(cudaMemcpy(h_val2_out, d_val2_in, N * sizeof(float2), cudaMemcpyDeviceToHost));

	cudaCHECK(cudaFree(d_key_in));
	cudaCHECK(cudaFree(d_val1_in));
	cudaCHECK(cudaFree(d_val2_in));
	cudaCHECK(cudaFree(d_key_out));
	cudaCHECK(cudaFree(d_val1_out));
	cudaCHECK(cudaFree(d_val2_out));
}