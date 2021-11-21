import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------
# 线性无关多项式拟合函数：
def myLeastSq(x, fx, order):
    A = np.empty((order+1, order+1))
    for i in range(order+1):
        for j in range(order+1):
            A[i, j] = sum(x**(i+j))

    b = np.empty((order+1))
    for i in range(order+1):
        b[i] = sum((x**i) * fx)

    a = np.linalg.solve(A, b)

    # 计算拟合值
    sx = x.copy()
    for i in range(len(sx)):
        sx[i] = fitFunction(x[i],a)

    # 计算误差
    err = 0
    for i in range(len(fx)):
        err = (sum((fx-sx)**2))**(0.5)

    return a, err


# --------------------------------------------
# 函数，构造线性无关多项式
def fitFunction(x, a):
    s = 0
    for i in range(len(a)):
        s += (x**i)*a[i]
    return s


# --------------------------------------------
# 函数：正交多项式作最小二乘拟合
def orthogonalPolyFit(x, fx, x_c, order):
    nk = x.size
    nc = x_c.size
    phi = np.empty((order+1, nk))
    phi_f = np.empty((order+1, nc))
    # 初始化展开系数矩阵
    para_phi = np.zeros((order+1, order+1))
    # 0阶设置
    para_phi[0, 0] = 1 # 0阶系数
    phi[0, :] = np.ones((1, nk))
    phi_f[0, :] = np.ones((1, nc))
    a = np.empty((order+1,))
    a[0] = np.sum(fx * phi[0, :]) / np.sum(phi[0, :]**2)
    # 开始迭代
    for i in range(1, order+1):
        tmp = phi[i-1, :]**2
        alpha = np.sum(x * tmp)  / np.sum(tmp)
        phi[i, :] = (x-alpha) * phi[i-1, :]
        phi_f[i, :] = (x_c-alpha) * phi_f[i-1, :]
        # 系数展开相关
        pad_para_phi = np.pad(para_phi[i-1, :], (1, 0)) 
        kernal = np.array([1, -alpha]) # 定义卷积核
        for j in range(i+1):
            para_phi[i, j] = np.sum(kernal * pad_para_phi[j: j+2])
        # 阶次大于1时要考虑beta项
        if i > 1:
            beta = np.sum(tmp) / np.sum(phi[i-2, :]**2)
            phi[i, :] -= beta * phi[i-2, :]
            phi_f[i, :] -= beta * phi_f[i-2, :]
            para_phi[i, :] -= beta * para_phi[i-2, :]
        # 计算正交系数序列
        a[i] = np.sum(fx * phi[i, :]) / np.sum(phi[i, :]**2)
        # 得到对应的线性无关多项式系数序列，用来验证和非线性相关的多项式系数是否一致
        para = a @ para_phi
    # 计算拟合函数值
    sx = np.zeros((nc,))
    for i in range(nc):
        for j in range(order+1):
            sx[i] += a[j] * phi_f[j, i]
    # 返回
    return sx, para


# ------------------------------------------
# 线性无关多项式拟合函数(法方程微小扰动)：
def myLeastSqDist(x, fx, order):
    A = np.empty((order+1, order+1))
    for i in range(order+1):
        for j in range(order+1):
            A[i, j] = sum(x**(i+j))

    b = np.empty((order+1))
    for i in range(order+1):
        b[i] = sum((x**i) * fx)

    # 微小扰动
    A[order, 0] -= 1e-5
    b[order] += 2e-5

    a = np.linalg.solve(A, b)

    # 计算拟合值
    sx = x.copy()
    for i in range(len(sx)):
        sx[i] = fitFunction(x[i],a)

    # 计算误差
    err = 0
    for i in range(len(fx)):
        err = (sum((fx-sx)**2))**(0.5)

    return a, err