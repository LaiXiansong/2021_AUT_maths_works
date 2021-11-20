# coding: utf-8
# # LU分解法
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as li


# 自己写的消去函数，太复杂了！################################################################
def LUDecomp_lxs(A, method):
    # 方阵大小
    n = A.shape[0]
    U = A.copy()
    L = np.eye(n)
    # 顺序消去过程
    for i in range(n):
        # 第i行i列对角元
        a = U[i, i]
        # 扫描以下的所有行元素
        for j in range(i + 1, n):
            lu1 = 0
            for k in range(i):
                lu1 += L[j, k] * U[k, i]
            L[j, i] = (A[j, i] - lu1) / a
            lu2 = 0
            for k in range(i + 1):
                lu2 += L[i + 1, k] * U[k, j]
            U[i + 1, j] = A[i + 1, j] - lu2
        # 输出
        # LU分解
    for i in range(1, n):
        for j in range(i):
            U[i, j] = 0
    if method == 'LU':
        return L, U
    # LDU分解
    if method == 'LDU':
        D = np.zeros((n, n))
        for i in range(n):
            # 提取对角元
            D[i, i] = U[i, i]
            # U的每行元素除以该行对角元
            U[i, :] = U[i, :] / U[i, i]
        return L, D, U


########################################################################################
# 示例标准代码，采用的思路是高斯消去
def LUDecomp(A, method):
    # 方阵大小
    n = A.shape[0]
    U = A.copy()
    L = np.eye(n)
    # 顺序消去过程
    for i in range(n):
        # 第i行i列对角元
        a = U[i, i]
        # 扫描以下的所有行元素
        for j in range(i + 1, n):
            # 输出
            # LU分解
            l = -U[j, i] / a
            U[j, :] = U[j, :] + l * U[i, :]
            # Uji强制置零，避免由于有限的精度，还存在一个很小的数
            U[j, i] = 0
            L[j, i] = -l
    if method == 'LU':
        return L, U
    # LDU分解
    if method == 'LDU':
        D = np.zeros((n, n))
        for i in range(n):
            # 提取对角元
            D[i, i] = U[i, i]
            # U的每行元素除以该行对角元
            U[i, :] = U[i, :] / U[i, i]
    return L, D, U


# 函数：生成Hilbert矩阵
def geneHilbMat(n):
    Hilb = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            Hilb[i, j] = 1 / (i + j + 1)
    return Hilb


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
test = geneHilbMat(5)
incvTest = invLowTriMat(test)
print(test)


# 函数：LU分解法求逆
def invLU(A):
    LA, UA = LUDecomp(A, 'LU')
    LA_inv = invLowTriMat(LA)
    UA_inv = np.transpose(invLowTriMat(np.transpose(UA)))
    invA = UA_inv @ LA_inv
    return invA


# 函数：矩阵$\infty$范数
def infNorm(A):
    n = A.shape[0]
    a = 0
    for i in range(n):
        if a < sum(abs(A[i, :])):
            a = sum(abs(A[i, :]))
    return a


# 函数：矩阵1范数
def oneNorm(A):
    n = A.shape[0]
    a = 0
    for i in range(n):
        if a < sum(abs(A[:, i])):
            a = sum(abs(A[:, i]))
    return a


# 函数：计算矩阵的条件数（无穷范数）
def cond(A):
    invA = invLU(A)
    return infNorm(A) * infNorm(invA)


# 计算不同阶次Hilbert矩阵的条件数
x = np.arange(10)
y = np.zeros(10, )
for i in range(1, 11):
    hilb = geneHilbMat(i)
    print(i, cond(hilb))
    y[i - 1] = cond(hilb)
plt.plot(x, y)
plt.show()
print()

# 随着Hilbert矩阵阶次的增加，条件数不断增大，到10阶矩阵的时候，条件数直接增长到$2e^7$。随着阶数增大，条件数的增大趋势呈指数级。

# ## 用Hilbert矩阵验证条件数和扰动的关系

# 上周的高斯消去函数
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


# 生成线性方程组
# 先生成一个三阶的Hilbert矩阵，条件数比较小
A_hilb3 = geneHilbMat(3)
x_ture3 = np.random.randn(3)
b_3 = np.inner(A_hilb3, x_ture3)
x_slove3 = li.solve(A_hilb3, b_3)
x_GausSolve3 = GaussianSolve(A_hilb3, b_3)
print(x_ture3, x_slove3, x_GausSolve3)

# 微小扰动
A_hilb3[2, 0] -= 0.000001
b_3[2] += 0.000002

# 验证结果
x_slove3 = li.solve(A_hilb3, b_3)
x_GausSolve3 = GaussianSolve(A_hilb3, b_3)
print(x_ture3, x_slove3, x_GausSolve3)

# 在条件数比较小的情况下，微小扰动导致了结果发生较小的变化。

# 生成线性方程组
# 生成一个十阶的Hilbert矩阵，条件数比较大
A_hilb10 = geneHilbMat(10)
x_ture10 = np.random.randn(10)
b_10 = np.inner(A_hilb10, x_ture10)
x_slove10 = li.solve(A_hilb10, b_10)
x_GausSolve10 = GaussianSolve(A_hilb10, b_10)
print('无扰动情况下：\nx的真值：\n', x_ture10, '\nx的sci库解\n', x_slove10, '\nx的高斯消去解\n', x_GausSolve10)

# 微小扰动
A_hilb10[9, 0] -= 0.000001
b_10[9] += 0.000002

# 验证结果
x_slove10 = li.solve(A_hilb10, b_10)
x_GausSolve10 = GaussianSolve(A_hilb10, b_10)
print('有扰动情况下：\nx的真值：\n', x_ture10, '\nx的sci库解\n', x_slove10, '\nx的高斯消去解\n', x_GausSolve10)
print()
# 在条件数比较大的情况下，微小扰动导致了结果发生很大的变化，验证了条件数对矩阵病态程度的反映情况。
#
