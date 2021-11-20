#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as li

# own lib
import basicLinearAlgebra as bLA
import LAE_DirectSolution as DS
import LAE_ItertionSolution as IS
import geneSpecialMat as SM


for n in [10, 20, 40]:
    A = SM.geneDlgMat([1, -8, 20, -8, 1], n)
    b = np.zeros(n)
    xOld = np.ones(n)
    xTrue = li.solve(A, b)
    s = -6  # 选择精度
    method = 'inf'
    itNumJ = 0
    eJ = IS.iterNorm(np.zeros(n), xOld, method)  # 初始值

    B, f = IS.BJandfJ(A, b)  # J法收敛次数
    while eJ > 10 ** s:
        xNewJ = B @ xOld + f
        eJ = IS.iterNorm(xOld, xNewJ, method)
        xOld = xNewJ
        itNumJ += 1
    print('J法>>', n, '阶5对角的迭代次数：', itNumJ)

    w = np.arange(0.5, 2, 0.05)
    k = np.zeros(len(w))
    RB_Lw_inf = np.zeros(len(w))
    for i in range(len(w)):
        xOld = np.ones(n)
        itNumJ = 0
        eJ = IS.iterNorm(np.zeros(n), xOld, method)  # 初始值
        B, f = IS.SOR(A, b, w[i])
        RB_Lw_inf[i] = -np.log(np.linalg.norm(B, np.inf))
        while eJ > 10 ** s:
            xNewJ = B @ xOld + f
            eJ = IS.iterNorm(xOld, xNewJ, method)
            xOld = xNewJ
            itNumJ += 1
        # eJFinal = IS.iterNorm(xTrue, xNewJ, method)  # 最终误差
        print('SOR法>>w==', w[i], n, '阶5对角的迭代次数：', itNumJ)
        k[i] = itNumJ

    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    plt.subplot(1, 2, 1)
    plt.plot(w, k)
    plt.xlabel('松弛因子')
    plt.ylabel('迭代次数k')
    plt.subplot(1, 2, 2)
    plt.plot(w, RB_Lw_inf)
    plt.xlabel('松弛因子')
    plt.ylabel('渐进收敛速度R(B)')
    plt.show()