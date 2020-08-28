import autograd.numpy as np
import matplotlib.pyplot as plt
import copy
from scipy.integrate import solve_ivp

from neuralODEs.neuralODE import neuralODE

def f_true(t,x): #ground truth ode
    A = - 1*np.array([[0.1,1],[1,0.1]])
    y = A@x
    return y

def lin_layer(x, w):  #linear ode
    return w@x

def tanh_layer(x, w): #single tanh layer ode
    a = w[:,:-1].T @ x
    y = np.tanh(a+ w[:,-1])
    return y

def one_hl(x, w):  #single hiddenlayer neural ode
    w1   = w[0:2,:]
    w2   = w[2:4, :]
    b1   = w[4,:]
    b2   = w[5,0:2]

    h = np.tanh(w1.T@x+b1)
    y = w2@h+b2
    return y

if __name__ == "__main__":

    M = 11 #number of time stencils
    t0, t1 = 0, 1 #start and endpoint
    x0s = np.random.normal(0, 1, (1000,2))  # 1000 random points in 2D

    #neuralODE
    D_in,K = 2 ,10 #D_in is ODE dim, K is dimension of hidden layer
    weights =  np.random.normal(0,0.1,size=(2*D_in+2,K))
    f = lambda x,w: one_hl(x,w)

    #linearODE #for linearODE just comment out
   # weights = np.random.normal(0, 0.1, size=(2,2))
   # f = lambda x, w: lin_layer(x, w)

    x1s = copy.copy(x0s)
    for k in range(0, len(x0s)):
        sol = solve_ivp(f_true, [t0, t1], x0s[k, :].flatten())
        x = sol.y[:, -1]
        x1s[k, :] = x

    node = neuralODE(f, weights, x0s, x1s, t0, t1, M)
    for i in range(0,1000):
        loss = node.train_step() #stochastic gradient descent
        # gt solution
        sol = solve_ivp(f_true, [node.t0, node.t1], y0 = x0s[-1,:].flatten(),t_eval=node.t_z)

        if i % 10 ==0:
            print(i)
            plt.clf()
            plt.plot(node.t_z,node.z.T[:, 0],'r',label='neuralODE')
            plt.plot(node.t_z,sol.y[0,:],'b',label='ground truth')
            plt.plot(node.t_a, node.a.T[:, 1], 'g' ,label='adjoint')
            plt.plot(node.t_z, node.z.T[:, 1],'r')
            plt.plot(node.t_z, sol.y[1, :],'b')
            plt.plot(node.t_a, node.a.T[:, 0], 'g')
            plt.legend(loc="upper left")
            plt.xlabel('time')
            plt.ylabel('state')
            plt.show()
            print(loss)
