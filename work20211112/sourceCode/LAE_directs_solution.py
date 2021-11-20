# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as li


##################################################################
# 高斯消去64
def GaussianSolve(A, b):
    n = len(b)
    Ab = np.c_[A, b]
    for i in range(n):
        a = Ab[i, i]
        for j in range(i + 1, n):
            k = -Ab[j, i] / a
            Ab[j:j + 1] += k * Ab[i:i + 1]
    x = np.zeros(n)
    x[n - 1] = Ab[n - 1, n] / Ab[n - 1, n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = Ab[i, n]
        for j in range(n - 1, i, -1):
            x[i] -= x[j] * Ab[i, j]
        x[i] /= Ab[i, i]
    return x
##################################################################


##################################################################
# 高斯消去16
# def GaussianSolve(A, b):
#     n = len(b)
#     Ab = np.c_[Mat2, b]
#     for i in range(n):
#         a = Ab[i, i]
#         for j in range(i + 1, n):
#             k = np.float16(-Ab[j, i] / a)
#             Ab[j:j + 1] += np.float16(k * Ab[i:i + 1])
#     x = np.zeros(n)
#     x[n - 1] = np.float16(Ab[n - 1, n] / Ab[n - 1, n - 1])
#     for i in range(n - 2, -1, -1):
#         x[i] = Ab[i, n]
#         for j in range(n - 1, i, -1):
#             x[i] -= np.float16(x[j] * Ab[i, j])
#         x[i] /= np.float16(Ab[i, i])
#     return x
##################################################################


##################################################################
# LU分解示例标准代码，采用的思路是高斯消去
def LUDecomp(A, method):
    n = A.shape[0]  # 方阵大小
    U = A.copy()
    L = np.eye(n)
    for i in range(n):
        a = U[i, i]  # 第i行i列对角元
        for j in range(i + 1, n):  # 扫描以下的所有行元素
            l = -U[j, i] / a
            U[j, :] = U[j, :] + l * U[i, :]  # Uji强制置零，避免由于有限的精度，还存在一个很小的数
            U[j, i] = 0
            L[j, i] = -l
    if method == 'LU':
        return L, U
    # LDU分解
    if method == 'LDU':
        D = np.zeros((n, n))
        for i in range(n):
            D[i, i] = U[i, i]  # 提取对角元
            U[i, :] = U[i, :] / U[i, i]  # U的每行元素除以该行对角元
        return L, D, U
##################################################################


##################################################################
# 函数：计算下三角矩阵的逆
def invLowTriMat(L):
    # 方阵大小
    n = L.shape[0]
    invL = np.zeros((n, n))
    # 从上开始扫描每一行
    for i in range(n):
        # 对角元
        invL[i, i] = 1 / L[i, i]
        # 扫描每一行，从对角元最邻近的元素开始至0列
        for j in range(i - 1, -1, -1):
            l = sum(invL[i, :] * L[:, j])
            invL[i, j] = -l / L[j, j]
    return invL
##################################################################


##################################################################
# 函数：LU分解法求逆
def invLU(A):
    LA, UA = LUDecomp(A, 'LU')
    LA_inv = invLowTriMat(LA)
    UA_inv = np.transpose(invLowTriMat(np.transpose(UA)))
    invA = UA_inv @ LA_inv
    return invA
##################################################################


##################################################################
# 函数：矩阵$\infty$范数
def infNorm(A):
    n = A.shape[0]
    a = 0
    for i in range(n):
        if a < sum(abs(A[i, :])):
            a = sum(abs(A[i, :]))
    return a
##################################################################


##################################################################
# 函数：矩阵1范数
def oneNorm(A):
    n = A.shape[0]
    a = 0
    for i in range(n):
        if a < sum(abs(A[:, i])):
            a = sum(abs(A[:, i]))
    return a
##################################################################


##################################################################
# 函数：计算矩阵的条件数（无穷范数）
def cond(A):
    invA = invLU(A)
    return infNorm(A) * infNorm(invA)
##################################################################

##################################################################
# 函数：A为三对角矩阵求解LAE,追赶法
def solveTridiagA(A, b):
    n = A.shape[0]
    [L, U] = LUDecomp(A, 'LU')
    # 求y
    y = np.zeros((n,))
    y[0] = b[0]
    for i in range(1, n):
        y[i] = b[i]-L[i, i-1]*y[i-1]
    # 求x
    x = np.zeros((n,))
    x[n-1] = y[n-1]/U[n-1, n-1]
    for i in range(n-2, -1, -1):
        x[i] = (y[i]-U[i, i+1]*x[i+1])/U[i, i]
    return x
##################################################################