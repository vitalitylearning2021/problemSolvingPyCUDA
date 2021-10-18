import numpy as np
import ctypes
from ctypes import * 

class FLOAT2(Structure):
    _fields_ = ("x", c_float), ("y", c_float)

def get_cuda_sort_by_key():
    dll = ctypes.windll.LoadLibrary("testSortByKeyDLL.dll") 
    func = dll.cudaSortByKey
    dll.cudaSortByKey.argtypes = [POINTER(c_int), POINTER(FLOAT2), POINTER(c_int), POINTER(FLOAT2), c_size_t] 
    return func

__cuda_sort_by_key = get_cuda_sort_by_key()

if __name__ == '__main__':
	
    N = 8

    test_key_in     = np.random.randint(10, size = N)
    test_key_out    = np.zeros((N,), dtype = np.int32)
    test_val_in     = np.random.rand(N, 2).astype(np.float32)
    test_val_out    = np.zeros((N, 2), dtype = np.float32)

    test_key_in_p   = test_key_in.ctypes.data_as(POINTER(c_int))
    test_key_out_p  = test_key_out.ctypes.data_as(POINTER(c_int))
    test_val_in_p   = test_val_in.ctypes.data_as(POINTER(FLOAT2))
    test_val_out_p  = test_val_out.ctypes.data_as(POINTER(FLOAT2))

    __cuda_sort_by_key(test_key_in_p, test_val_in_p, test_key_out_p, test_val_out_p, N)

    print(test_key_in)
    print(test_key_out)
    print(test_val_in)
    print(test_val_out)
