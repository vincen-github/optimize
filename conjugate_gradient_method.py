from torch import autograd
from torch import tensor
from torch import cat
from numpy.linalg import norm
class conjugate_gradient_method(object):
    """
    conjugate_gradient_method object:

        The class which to solve the optimal problem using  conjugate gradient method.
        in this class.you only need to build and pass the object function.
        But it is worth noting that you can't use numpy when you build the object function.
        Here is an example for reference.
            func = lambda x : (x[0] - 4)**2 + (x[1] + 2)**2 + 1
        Using the "**" to represent the power instead of numpy.power().
        Or you can build the tensor form by torch.

        parameters:
            1. func: the object function.
            2. x0 : initial point, the dtype of the elements in it must be float.
            3. eps : the iteration is broken if the norm of gradient is less than eps.

    """
    def __init__(self, func, x0, eps):
        self.func = func
        self.x0 = x0
        self.eps = eps

    def derivative(self, x):
        x = tensor(x, requires_grad = True)
        y = self.func(x)
        grad = autograd.grad(y, x,
                            retain_graph = True,
                            create_graph = True)[0]
        
        hessian = tensor([])
        for anygrad in grad:
            hessian = cat((hessian,
                        autograd.grad(anygrad, 
                                    x, 
                                    retain_graph = True, 
                                    allow_unused = True)[0]
                        ))

        return grad.detach().numpy(), hessian.detach().numpy().reshape(grad.shape[0], -1)

    def calculate_direction(self, x, direction):
        if self.method == "FR":
            # 计算当前点的梯度与hessian矩阵
            grad, hessian = self.derivative(x)
            # direction = -grad + beta*direction
            # where beta = (direction@hessian@grad)/(direction@hessian@direction)
            beta = (direction@hessian@grad)/(direction@hessian@direction)
            direction = -grad + beta*direction

            return direction


    def calculate_step(self, x, direction):
        if self.method == "FR":
            # 先计算gradient与hessian矩阵
            grad, hessian = self.derivative(x)

            return -(direction@grad)/(direction@hessian@direction)

    def run(self, method = 'FR'):
        method_tuple = tuple(['FR'])
        if method not in method_tuple:
            raise Exception("please check the method string.")
        # 将method设置为类属性
        self.method = method
        # 设置迭代点为初始点
        x = self.x0
        # 若传入的迭代方式为FR
        if self.method == 'FR':
            # 计算初始点的梯度
            grad, _ = self.derivative(x)
            # 设置当前更新方向为负梯度
            direction = -grad
            # 若梯度的范数小于eps,直接输出初始点
            if norm(grad) < self.eps:
                # 输出初始点
                return x
            # 否则进行第一次最优点的更新
            # 利用FR公式计算步长
            step = self.calculate_step(x, direction)
            # 进行最优点的更新
            x = x + step*direction
            # 计算新的迭代点处的grad
            grad, _ = self.derivative(x)

            # 梯度范数小于eps,停止迭代
            while norm(grad) > self.eps:
            #     # 计算新的迭代方向
                direction = self.calculate_direction(x, direction)
                #计算步长
                step = self.calculate_step(x, direction)
                # 更新
                x = x + step*direction
            return x

if __name__ == "__main__":
    # the parameters of optimal problem
    func = lambda x : (x[0] - 3)**2 + x[1]**2
    x0 = ([-2., 6.])
    eps = .1

    # test
    solution = conjugate_gradient_method(func, x0, eps)

    print(solution.run(method = 'FR'))

