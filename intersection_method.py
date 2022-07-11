# -*- coding: utf-8 -*-
"""
Created on Sat May 14 10:25:09 2022

@author: mcamboim
"""
import numpy as np

def left_pinv(A : np.array) -> np.array: 
    return np.linalg.inv(A.T @ A) @ A.T

def right_pinv(A : np.array) -> np.array:
    return A.T @ np.linalg.inv(A @ A.T)

class intersection_method():
    
    def __init__(self):
        pass
        
    def __hankel(self, data : np.array, init_step : float, lines : float, columns : float) -> np.array:
        data_size = data.shape[1]
        H = np.zeros((lines*data_size,columns))
        
        for line in range(lines):
            for column in range(columns):
                start_line_idx = line * data_size
                stop_line_idx = data_size * (line  + 1)
                start_column_idx = column
                stop_column_idx = column + 1
                step = init_step + column + line
                H[start_line_idx:stop_line_idx, start_column_idx:stop_column_idx] = data[step,:].reshape(-1,1)
        
        return H
    
    
    def __ioHankel(self, uk : np.array, yk : np.array, lines : float, columns : float) -> tuple:
        Up = self.__hankel(data = uk, init_step = 0, lines = lines, columns = columns)
        Uf = self.__hankel(data = uk, init_step = lines, lines = lines, columns = columns)
        Yp = self.__hankel(data = yk, init_step = 0, lines = lines, columns = columns)
        Yf = self.__hankel(data = yk, init_step = lines, lines = lines, columns = columns)
        
        return Up,Uf,Yp,Yf
    
    
    def __findW(self, Up : np.array, Uf : np.array, Yp : np.array, Yf : np.array, lines : float, columns : float, input_size : float, output_size : float) -> tuple:
        W = np.zeros((2*(input_size+output_size)*lines,columns))
        l1 = input_size*lines
        l2 = l1 + output_size*lines
        l3 = l2 + input_size*lines
        l4 = l3 + output_size*lines
        
        W[0:l1,:] = np.copy(Up)
        W[l1:l2,:] = np.copy(Yp)
        W[l2:l3,:] = np.copy(Uf)
        W[l3:l4,:] = np.copy(Yf)
        Wp = np.copy(W[0:l2,:])
        Wf = np.copy(W[l2:l4,:])
        
        return W, Wp, Wf
    
    def __findSSM(self, Uf : np.array, Yf : np.array, Xf : np.array, input_size : float, output_size : float) -> tuple:
        X_plus = np.copy(Xf[:,1:])
        X = np.copy(Xf[:,:-1])
        Y = np.copy(Yf[:output_size,:-1])
        U = np.copy(Uf[:input_size,:-1])
        
        X1Y_lines = X_plus.shape[0] + Y.shape[0]
        X1Y_columns = Y.shape[1]
        X1Y = np.zeros((X1Y_lines,X1Y_columns))
        X1Y[:X_plus.shape[0],:] = np.copy(X_plus)
        X1Y[X_plus.shape[0]:,:] = np.copy(Y)
        
        XU_lines = X.shape[0] + U.shape[0]
        XU_columns = U.shape[1]
        XU = np.zeros((XU_lines,XU_columns))
        XU[:X.shape[0],:] = np.copy(X)
        XU[X.shape[0]:,:] = np.copy(U)        
        
        n = X.shape[0]
        temp = X1Y @ right_pinv(XU)
        A_hat = temp[:n,:n]
        B_hat = temp[:n,n:]
        C_hat = temp[n:,:n]
        D_hat = temp[:n,:n]
        
        return A_hat,B_hat,C_hat,D_hat
    
    
    def intersectionMethod(self, uk : np.array, yk : np.array, lines : float, columns : float) -> tuple:
        input_size = uk.shape[1]
        output_size = yk.shape[1]
        Up,Uf,Yp,Yf = self.__ioHankel(uk = uk, yk = yk, lines = lines, columns = columns)
        W,Wp,Wf = self.__findW(Up = Up, Uf = Uf, Yp = Yp, Yf = Yf, lines = lines, columns = columns, input_size = input_size, output_size = output_size)
        
        U,S,VT = np.linalg.svd(W)
        S11 = np.diag(S[S>1e-5])
        singular_values = S11.shape[0]
        u11_lines = int(W.shape[0] / 2)
        U11 = np.copy(U[:u11_lines,:singular_values])
        U12 = np.copy(U[:u11_lines,singular_values:])
        
        U,S,VT = np.linalg.svd(U12.T @ U11 @ S11)
        Sq = np.diag(S[S>1e-5])
        singular_values = Sq.shape[0]
        Uq = np.copy(U[:,:singular_values])
        
        Xf = Uq.T @ U12.T @ Wp
        
        A_hat,B_hat,C_hat,D_hat = self.__findSSM(Uf = Uf, Yf = Yf, Xf = Xf, input_size = input_size, output_size = output_size)
        
        return A_hat,B_hat,C_hat,D_hat
    
    
    
    
    

    