import numpy as np
import ctypes
from ctypes import * 

def get_cuda_sort():
    dll = ctypes.windll.LoadLibrary("testSortDLL.dll") 
    func = dll.cudaSort
    dll.cudaSort.argtypes = [POINTER(c_int), POINTER(c_int), c_size_t] 
    return func

__cuda_sort = get_cuda_sort()

if __name__ == '__main__':
	
    N = 8

    test_in     = np.random.randint(10, size = N)
    test_out    = np.zeros((N,), dtype = np.int32)

    test_in_p   = test_in.ctypes.data_as(POINTER(c_int))
    test_out_p  = test_out.ctypes.data_as(POINTER(c_int))

    __cuda_sort(test_in_p, test_out_p, N)

    print(test_in)
    print(test_out)
