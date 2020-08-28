import autograd.numpy as np
from autograd import jacobian
from scipy.integrate import solve_ivp

class neuralODE():
    #TODO write asserts
    def __init__(self,fun,weights,x0s,x1s,t0,t1,M,delta = 0.01,iter_max=100, batch = 100):
        self.iter_max = iter_max
        self.batch    = batch
        self.fun = fun
        self.delta = delta
        self.w = weights
        self.x0s = x0s
        self.x1s = x1s
        self.x0 = x0s[0,:].flatten()
        self.x1 = x1s[0,:].flatten()
        self.t0 = t0
        self.t1 = t1
        self.M = M
        self.t = np.linspace(t0,t1,M)

        self.z = 0
        self.a = 0


    def fun_fwd(self,x,w):
        "Output: fun at x, w"
        "Input: fun input"
        y = self.fun(x,w)
        return y

    def jacobian(self, x, t, w):
        "Output: Jacobian of fun w.r.t x"
        "Input: fun input"

        t0 = self.t0
        t1 = self.t1
        M = self.M

        f = lambda x: self.fun(x,w=self.w) #fun at weights = w
        df_dx = lambda y: jacobian(f)(y)   #jacobian  of fun

        t_span = np.linspace(t0, t1, M)
        result = np.where((t - (t1-t0) / M <= t_span) * (t_span <= t + (t1-t0) / M))
        try:
            x_in = x[:,result[0][0]].flatten()
        except:
            x_in = x[:, 0].flatten()
        J = df_dx(x_in)
        return J

    def forward(self):
        "Output: Forward solution of ODE [z(0),...,z(T)]"
        x0 = self.x0
        weights = self.w

        def f(t,x):
            return self.fun_fwd(x,w=weights)

        tau = [self.t[0],self.t[-1]]
        sol= solve_ivp(f, tau, y0 = x0,t_eval =self.t)
        self.z =   sol.y
        self.t_z = sol.t
        return None

    def backward(self):
        "Output: Backward solution of adjoint ODE [a(T),...,a(0)]"
        def augmented(t,a):
            J = self.jacobian(self.z,t,self.w)
            return -a@J

        sol = solve_ivp(augmented, [self.t[-1], self.t[0]],y0 = self.dloss_dx(), t_eval=np.flip(self.t))
        self.a = sol.y
        self.t_a = sol.t
        return None

    def dloss_dw(self):
        "Output: Gradient of Loss dL_dw w.r.t weights"
        t0 = self.t0
        t1 = self.t1
        M = self.M

        df_dw = lambda x,w: jacobian(self.fun,1)(x,w)

        dloss_dw = 0
        for m in range(0,  M): #integration for dullies
            z_in = self.z[:, M-m-1].flatten()
            a_in = self.a[:, m].flatten()
            J = df_dw(z_in, self.w)
            l = a_in[0] * J[0] + a_in[1] * J[1]
            dloss_dw -= l*(t1-t0)/M
        return dloss_dw

    def loss(self):
        "Output: MSE loss"
        return np.sum(np.power(self.x1 - self.z[:, -1].flatten(), 2))

    def dloss_dx(self):
        "Output: Gradient of MSE loss"
        return 2*(self.x1 - self.z[:, -1].flatten())

    def train_step(self):
        "Perform single step of stochastic gradient descent"
        delta = self.delta

        w = self.w
        loss = 0
        batch = np.random.randint(0,len(self.x0s),self.batch)
        for k in batch :

            self.x0 = self.x0s[k, :].flatten()
            self.x1 = self.x1s[k, :].flatten()
            self.forward()
            self.backward()
            gradient_w = self.dloss_dw()

            w -= delta*gradient_w/len(batch) #update weights
            loss += np.sum(np.power(self.x1 - self.z[:, -1].flatten(), 2))/len(self.x0s)

        self.x0 = self.x0s[-1, :].flatten()
        self.x1 = self.x1s[-1, :].flatten()
        self.forward()
        self.backward()
        self.w = w
        return loss





