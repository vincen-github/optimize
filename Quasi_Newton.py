from torch import pow
from torch import tensor
from torch import autograd
from torch import is_tensor
from numpy import array
from numpy import eye
from numpy import ones
from numpy.linalg import norm
from numpy import float as np_float

class QuasiNewton(object):
    '''
    QuasiNewton
    -----------
    A commonly used optimization method.

        @author: vincen

        @github: https://github.com/vincen-github

        principle: 
            DFP: H(k+1) = H(k) - (H(k)@y(k)@y(k)@H(k))/(y(k)@H(k)@y(k)) + (s(k)@s(k))/(y(k)s(k))

    Available attributes
    ----------------------
        1. func:function
            On the objective function of tensors.

        2. x0: tensor or ndarray.
            Iteration initial point.

        3. eps: 'float', default = 1e-3
            Update termination threshold.

        4. method: 'str', default = 'DFP'
            Type of update formula.
    
    Available function
    -------------------
        1.optimize(self)
            Optimization function.
        
        2. gradient(self, x)
            Method of calculating the gradient of incoming points

            Parameters:
            ----------
            x: numpy or tensor
                The point which you want to calculate the gradient of objective function.
        
        3. one_dimensional_search(self, x, d)
            One-dimensional precise search for determining the step size.

            Parameters:
            -----------
            x:numpy or tensor
                The optimization point where one-dimensional precise search is located.
                
            d:numpy or tensor
                Optimization direction.

            max_iter: int
                The maximum number of iterations.
            
            eta: float
                learning rate.
    '''
    def __init__(self, func, x0, eps = 1e-3, method = 'DFP'):
        self.func = func
        self.x0 = x0
        self.eps = eps
        self.method = method

    def gradient(self, x):
        # 若传入点类型为ndarray, 转为tensor.
        if not is_tensor(x):
            x = tensor(x, requires_grad = True)
        # 建立图关系
        y = self.func(x)
        # 计算梯度
        grad = autograd.grad(y ,x)[0]

        return grad.detach().numpy()

    def one_dimensional_search(self, x, d, max_iter = 200, eta = 1e-2):
        # 为了使代码适用于各形式的函数,这里用gradient descent代替解析解
        if not is_tensor(x):
            x = tensor(x, requires_grad = False)
        if not is_tensor(d):
            d = tensor(d, requires_grad = False)
        # 初始化alpha = 0
        alpha = tensor(1.0, requires_grad = True)
        for loop in range(max_iter):
            # 计算函数值
            y = self.func(x + alpha*d)
            grad = autograd.grad(y, alpha, retain_graph = True)[0]
            alpha = alpha - eta*grad
            if alpha < 0:
                break

        return alpha.detach().numpy()
    
    def optimize(self):
        # 若选择更新方法为DFP
        if self.method == 'DFP':
            #初始化近似矩阵
            H = eye(self.x0.shape[0])
            # 计算初始点处的gradient
            grad = self.gradient(x0)
            # 初始化方向为负梯度方向
            d = -grad
            # 初始化迭代点
            x = self.x0.copy()
            # ||g|| < eps,跳出循环
            while(norm(grad) > self.eps):
                # pre_x代表前一轮的更新点
                pre_x = x
                # pre_grad代表前一轮的梯度
                pre_grad = grad
                alpha = self.one_dimensional_search(x, d)
                x = x + alpha*d
                # 更新梯度
                grad = self.gradient(x)
                s, y = (x - pre_x).reshape(-1, 1), (grad - pre_grad).reshape(-1, 1)
                H = H - (H@y@y.T@H)/(y.T@H@y) + (s@s.T)/(y.T@s)
                d = -H@grad
        
        return x

if __name__ == "__main__":
    func = lambda x: pow(x[0], 2) + 2*pow(x[1], 2) - 2*x[0]*x[1] - 4*x[0]
    x0 = array([1, 1], dtype = np_float)

    solution = QuasiNewton(func, x0)
    print(solution.optimize())
    print(solution.gradient(x0))
    

