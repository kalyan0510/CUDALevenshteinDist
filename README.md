# CUDALevenshteinDist
Implemented levenshtein distance on CUDA device

## 1.1 Simple implementation of ED using python arrays

This implementation uses simple python arrays/lists. The dp algorithm progressively fills the ED matrix and arrives at the solution. However, all the steps in this algo are highly sequential, taking up a lot of time in processing.

### 1.2 Array manipulation based Approach
The bottleneck of the previous implementation is the sequential dependency. An element in a particular row and column can only be computed if all elements in the earlier rows and columns were computed. 

Since in each step, we are performing multiple computations, like: 
1. incrementing upper matrix cells by 1
2. incrementing diagonally top-left cell by 1 depending on equality of corresponding characters  
3. incrementing the left cell elements

We can see that, in computing the current cell in a row, the sequential dependency only kicks in when reading value from left. So, we can isolate this part of computation until necessary, while performing other calculations in parallel|bulk without any restricstions.
![download_1](https://user-images.githubusercontent.com/14043633/191613103-360beb56-eaf1-4222-8c2f-eb68a3160eb7.png)
![download](https://user-images.githubusercontent.com/14043633/191613105-527f99ab-f458-4056-ba22-a423657a8e1b.png)

###1.3 GPU implementation of ED (computing diagonal wise)

![download](https://user-images.githubusercontent.com/14043633/191613562-6293f10e-4e5a-4f8a-b0f5-63a82e5de3f9.png)

Row wise computation vs diagonal wise computation
```
>>> [0 1 2 3 4 5 6 7 8 9]
>>> [1 0 1 2 3 4 5 6 7 8]
>>> [2 1 1 2 3 3 4 5 6 7]
>>> [3 2 1 1 2 3 3 4 5 6]
>>> [4 3 2 1 2 3 3 4 5 6]
>>> [5 4 3 2 2 3 4 4 4 5]
>>> [6 5 4 3 3 2 3 4 5 5]
>>> [7 6 5 4 4 3 3 3 4 5]
>>> [8 7 6 5 5 4 4 4 3 4]
[8 7 6 5 5 4 4 4 3 4]
>>>         [0]
>>>        [1 1]
>>>       [2 0 2]
>>>      [3 1 1 3]
>>>     [4 2 1 2 4]
>>>    [5 3 1 2 3 5]
>>>   [6 4 2 1 3 4 6]
>>>  [7 5 3 1 2 3 5 7]
>>> [8 6 4 2 2 3 4 6 8]
>>> [7 5 3 2 3 3 5 7 9]
>>>  [6 4 3 3 3 4 6 8]
>>>   [5 4 2 4 4 5 7]
>>>    [5 3 3 4 5 6]
>>>     [4 3 4 4 6]
>>>      [4 3 5 5]
>>>       [4 4 5]
>>>        [3 5]
>>>         [4]
[8 7 6 5 5 4 4 4 3
```

In the GPU implementation the diagonals are computed one after the other. But the entire diagonal can be computed at once, which is implemented using CUDA kernel.

### Analysis
With NVIDIA Tesla T4, the CUDA implementaion is much faster than the CPU implementations:  
The graphs is in the log scale:  

![download](https://user-images.githubusercontent.com/14043633/191614849-32a8b1fa-4a18-43ca-a600-637e5c65a2f8.png)

