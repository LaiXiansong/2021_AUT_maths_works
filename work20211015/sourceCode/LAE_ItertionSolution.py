#!/usr/bin/env python
# coding: utf-8

# # Jocobi迭代法和Gauss-Seidel迭代法


import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as li


##################################################################
# 求迭代用到的LUD矩阵
def Mat2LUD(A):
    n = A.shape[0]
    D = np.zeros((n, n))
    L = np.zeros((n, n))
    U = np.zeros((n, n))
    for i in range(n):
        D[i, i] = A[i, i]
        idx1 = np.arange(0,i)
        idx2 = np.arange(i+1,n)
        L[i, idx1] = -A[i, idx1]
        U[i, idx2] = -A[i, idx2]
    return L, U, D
##################################################################


##################################################################
# 找BJ和fJ
def BJandfJ(Mat_A, b):
    n = Mat_A.shape[0]
    E = np.eye(n)
    Mat_L, Mat_U, Mat_D = Mat2LUD(Mat_A)
    BJ = E-(np.linalg.inv(Mat_D))@(Mat_A)
    fJ = np.linalg.inv(Mat_D)@b
    return BJ, fJ
##################################################################


##################################################################
# 找BG和fG
def BGandfG(A, b):
    n = A.shape[0]
    E = np.eye(n)
    Mat_L, Mat_U, Mat_D = Mat2LUD(A)
    BG = (Mat_D-Mat_L) @ (Mat_U)
    fG = np.linalg.inv(Mat_D-Mat_L) @ b
    return BG, fG
##################################################################


##################################################################
# 迭代收敛范数准则(向量范数)
def iterNorm(xOld, xNew, method):
    if method == '1':
        e = np.sum(np.abs(xNew - xOld))
    elif method == '2':
        e = np.sum(xNew - xOld)**2
    elif method == 'inf':
        e = np.max(np.abs(xNew - xOld))
    return e
##################################################################


##################################################################
# SOR法迭代
def SOR(A, b, omiga):  # 迭代因子omiga
    n = A.shape[0]
    E = np.eye(n)
    A_L, A_U, A_D = Mat2LUD(A)
    Lw_B = (np.linalg.inv(A_D-omiga*A_L)) @ (omiga*A_U+(1-omiga)*A_D)
    Lw_f = omiga*np.linalg.inv(A_D-omiga*A_L) @ b
    return Lw_B, Lw_f
##################################################################