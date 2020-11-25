from numpy.linalg import inv
from numpy.linalg import norm
from torch import tensor
from torch import cat
from torch import autograd

class newton_method(object):
    """
    newton_method object:
        The class which to find the minimal point of the object function.
        In this class, you don't need to build and pass the function which to calculate gradient and hessian matrix by yourself.
        i use the automatic derivation mechanism encapsulated in pytorch to solve it.
        parameters:
            func: the object function
            x0 : initial point
            eps : the iteration is broken if the norm of gradient is less than eps.
    """
    def __init__(self, func, x0, eps = 1e-2):
        self.func = func
        self.x = x0
        self.eps = eps

    def derivative(self, x, order = 2):
        x = tensor(x, requires_grad = True)
        y = func(x)

        if order == 2:
            grad = autograd.grad(y ,x, retain_graph = True, create_graph = True)[0]
            # 若order为2,则求二阶导
            hessian = tensor([])
            for anygrad in grad:
                hessian = cat((hessian, autograd.grad(anygrad, x, retain_graph = True, allow_unused = True)[0]))

            return grad.detach().numpy(), hessian.detach().numpy().reshape(grad.shape[0], -1)

        else:
            grad = autograd.grad(y , x, retain_graph = True, create_graph = True)[0]
            return grad.detach().numpy()
    
    def run(self):
        grad =  self.derivative(self.x ,1) 
        while norm(grad) > self.eps:
            # calcuate the direction in point x
            grad, hessian = self.derivative(self.x)
            d = -inv(hessian)@grad
            self.x = self.x + d

        return self.x
    

if __name__ == "__main__":
    func = lambda x : (x[0] - 4)**2 + (x[1] + 2)**2 + 1
    x0 = [0., 0.]
    eps = .1
    

    solution = newton_method(func, x0, eps)
    print(solution.derivative(x0))
    print(solution.run())