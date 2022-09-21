import numpy as np
from numba import cuda
import math 
import time
from tqdm import tqdm
from google.colab import drive
import matplotlib.pyplot as plt

def ed_simple(X, Y):
    if len(Y) == 0:
        return list(range(len(X)))
    prev = list(range(len(Y) + 1))
    for i, c1 in enumerate(X):
        curr = [i + 1]
        for j, c2 in enumerate(Y):
            curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + (c1 != c2)))
        prev = curr
    return prev
    

def ed_np(X, Y, M=None):
    m, n = len(X), len(Y)
    if M is None:
        # create new array
        M = np.arange(n + 1)
    else:
        # else fill it inplace
        for i in range(n + 1):
            M[i] = i
    # print(">>>", M)
    for i, lchar in enumerate(X):
        tmp = M[:-1]+(lchar!=Y)
        M[1:] = np.minimum(M[1:]+1, tmp)
        M[0]=i+1
        xn = np.minimum.accumulate(M[1:]+np.arange(n-1, -1, -1)) - np.arange(n-1, -1, -1)
        M[1:] = np.minimum(xn, np.arange(i+1, i+n+1))
        # print(">>>", M)
    return M
    

@cuda.jit
def ed_kernel(X, Y, past, last, curr, cfg):
    # cfg = [si, sj, l, d] 
    x = cuda.grid(1)
    if x >= cfg[2]:
        return
    i, j = cfg[0]-x, cfg[1]+x # si+x, sj-x
    if i==0 or j==0:
        curr[x]=i+j
        return
    char_eq = Y[i-1]!=X[j-1]#0 if Y[i-1]==X[j-1] else 1
    if cfg[3]<=Y.shape[0]: # d<=n
        curr[x] = min(past[x-1]+char_eq, min(last[x-1], last[x])+1)
    elif cfg[3]<=Y.shape[0]+1: # d<=n+1
        curr[x] = min(past[x]+char_eq, min(last[x], last[x+1])+1)
    else: # else d
        curr[x] = min(past[x+1]+char_eq, min(last[x], last[x+1])+1)

def ed_gpu(X, Y, X_cuda=None, Y_cuda=None, TPB=1024, st=True):
    if X.shape[0] < Y.shape[0]: return ed_gpu(Y, X, Y_cuda, X_cuda, TPB, st=not st) 
    # print("ST",st)
    m,n = X.shape[0], Y.shape[0]
    X = X.view(np.int32)
    Y = Y.view(np.int32)
    # print(m,n)
    if X_cuda is None:
        X_cuda = cuda.to_device(X)
    if Y_cuda is None:
        Y_cuda = cuda.to_device(Y)
    
    past = cuda.device_array((Y.shape[0]+1), dtype=np.uint32) 
    last = cuda.device_array((Y.shape[0]+1), dtype=np.uint32) 
    curr = cuda.device_array((Y.shape[0]+1), dtype=np.uint32) 

    threadsperblock = TPB
    
    def get_diag_desc(diag_i):
        if d<=n:
            return (diag_i, 0), diag_i+1
        elif d<=m:
            return (n, diag_i-n), n+1
        else:
            return (n, diag_i-n), m+n+1-diag_i
    # tt=0
    last_row = []
    for d in range(X.shape[0]+Y.shape[0]+1):
        # st = time.time()
        blockspergrid = int(math.ceil((d+1) / threadsperblock))
        (si,sj),l = get_diag_desc(d)
        # cfg = np.array([si, sj, l, d])
        cfg = cuda.to_device(np.array([si, sj, l, d]))
        # tt =tt + time.time() - st   
        ed_kernel[blockspergrid, threadsperblock](X_cuda, Y_cuda, past, last, curr, cfg)

        # uncommeneting this line prints the diagonal lines in a diamond shape
        # print(">>>", (" "*((Y.shape[0]+1-l)))+str(curr.copy_to_host()[:l]))
        if st:
            if d>=m:
                last_row.append(curr[l-1])
        else:
            if d>=n:
                last_row.append(curr[0])
        past, last, curr =   last, curr, past
            

    # print(tt)
    return last_row
