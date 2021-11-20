import numpy as np


##################################################################
# 函数：生成n阶Hilbert矩阵
def geneHilbMat(n):
    Hilb = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            Hilb[i, j] = 1 / (i + j + 1)
    return Hilb
##################################################################


##################################################################
# 函数：生成多对角矩阵
def geneDlgMat(nums: list, n: int):
    ap = len(nums)
    if ap % 2 == 1:
        A = np.zeros((n, n + ap - 1))
        for i in range(n):
            for j in range(ap):
                A[i,i + j] = nums[j]
        A = A[:,int((ap-1)/2):n+int((ap-1)/2)]
        return A
    else:
        print('请输入一个奇数')
        return None
##################################################################