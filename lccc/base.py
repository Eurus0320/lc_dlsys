from __future__ import  absolute_import

import os
import ctypes
import numpy as np

float32 = np.float32
lib = np.ctypeslib.load_library("./src/main.so", ".")

def check_call(ret):
    assert(ret == 0)

def c_array(ctype, values):
    return (ctype * len(values))(*values)

def cast_to_ndarray(arr):
    if isinstance(arr, np.ndarray):
        return arr
    if not isinstance(arr, list):
        arr = [arr]
    return np.array(arr)