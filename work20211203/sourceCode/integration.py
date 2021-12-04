import numpy as np
import math

#--------------------------------------------------------
# cotes求积系数
def cotes(n, k):
    return ((-1)**(n-k)) / math.factorial(k) / math.factorial(n-k) / n * cotesInteg(n, k)

#--------------------------------------------------------
# cotes系数中的积分结果
def cotesInteg(n, k):
    ori_papa = np.zeros((n+3,)) # 经过padding
    temp_para = ori_papa.copy()
    inte_para = np.zeros((n+2,))
    ori_papa[1] = 1
    for j in range(0, n+1):
        if j != k:
            kenel = np.array([1, -j])
            for i in range(n+2):
                temp_para[i+1] = np.sum(ori_papa[i:i+2] * kenel)
            ori_papa = temp_para.copy()
    inte_para[0] = 0
    for i in range(1, n+2):
        inte_para[i] = ori_papa[i]/i
    return s(n, inte_para)

#-------------------------------------------------------
# 多项式
def s(x, coef):
    sx = 0
    for i in range(len(coef)):
        sx += (x**i) * coef[i]
    return sx

#-------------------------------------------------------
# Chebyshev
def gaussCheb(n, a, b, func):
    t = np.zeros((n+1, ))
    A = np.ones((n+1,)) * np.pi/(n+1)
    for k in range(n+1):
        t[k] = np.cos((2*k+1)*np.pi/2/(n+1))
    I = 0
    for k in range(n+1):
        I += np.sqrt(1-t[k]**2) * func((a+b)/2 + (b-a)/2*t[k])
    return I * (b-a)/2*np.pi/(n+1)

#-------------------------------------------------------
# 复合Gauss-chebyshev
# combined_Chebyshev
def combGaussCheb(m, n, a, b, func):
    x = np.linspace(a, b, m+1, endpoint=True)
    I_plus = 0
    for k in range(m):
        I_plus += gaussCheb(n, x[k], x[k+1], func)
    return I_plus


#-------------------------------------------------------
# 复合梯形
def combTrap(n, a, b, func):
    h = (b - a) / n
    x = np.linspace(a, b, n+1, endpoint=True)
    cs = func(a) + func(b)
    for k in range(1, n):
        cs += 2 * func(x[k])
    return cs * h/2

#-------------------------------------------------------
# Romberg
def Romberg(p, a, b, func):
    # p: Romberg的层数
    T = np.zeros((p, p))
    for i in range(p):
        T[i, 0] = combTrap(2**i, a, b, func)
        for k in range(1, i+1):
            T[i, k] = (4**k * T[i, k-1] - T[i-1, k-1]) / (4**k - 1)
    return T