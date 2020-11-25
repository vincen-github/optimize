class advance_retreat(object):
    '''
    advance_retreat object:
        parameters:
            func: object function in optimal problem
            x0: initial point
            h: initial step
            alpha: the gain of step
    '''
    def __init__(self, func, x0, h, alpha = 2):
        self.func = func
        self.x1 = x0
        self.x2 = x0 + h
        # x3与x4的存在的意义是为了最终得到的搜索区间长度更短
        self.x3 = 0
        self.x4 = 0

        # set the step
        self.h = h
        # iterate number
        self.k = 1
        self.alpha = alpha
        # get the value of function
        self.y1 = self.func(self.x1)
        self.y2 = self.func(self.x2)
        # y3, y4是x3, x4处的函数值
        # 初始定位0
        self.y3 = 0
        self.y4 = 0
    def search(self):
        if self.y1 < self.y2:
            # 若第一次前进处的函数值大于初始点的函数值
            # 最终输出的搜索区间右端点确定为x2
            # 左端点的确定从初始点开始后退,
            # 直到后退至self.y1 > self.y2
            # 后退步长设置为k*alpha
            while(self.y1 < self.y2):
                self.x1  = self.x1 - self.k*self.alpha*self.h
                self.y1 = self.func(self.x1)
                # 这里对x2做一个小的修正,与x1一起后退
                # 原则是x2的函数值y2保持大于y1
                # 将x2移动后的函数值用x3暂存，判断y3其大于y1时,将x3赋值给x2
                self.x3 = self.x2 - self.k*self.alpha*self.h
                self.y4 = self.func(self.x3)
                if self.y3 > self.y2:
                    self.x2 = self.x3
                    self.y2 = self.y3

                k += 1
            return self.x1, self.x2

        elif self.y1 > self.y2:
            # 若第一次前进处的函数值大于初始点的函数值
            # 最终输出的搜索区间左端点端点确定为x1
            # 左端点的确定从x2开始前进,
            # 直到前进至self.y2 > self.y1
            # 前进步长设置为k*alpha
            while(self.y1 > self.y2):
                self.x2 = self.x2 + self.k*self.alpha*self.h
                self.y2 = self.func(self.x2)
                # 这里对x1做一个小的修正,与x2一起前进
                # 原则是x1的函数值y1保持大于y2
                # 将x1移动后的函数值用x4暂存，判断y3其大于y2时,将x4赋值给x1
                self.x4 = self.x1 + self.k*self.alpha*self.h
                self.y4 = self.func(self.x4)
                if self.y4 > self.y2:
                    self.x1 = self.x4
                    self.y1 = self.y4

                self.k += 1 
            return self.x1, self.x2
        else:
            # 若x1处的函数值等于x2处的函数值
            # 对于单峰函数而言,搜索区间即为(x1, x2)
            return self.x1, self.x2
        


if __name__ == "__main__":
    from numpy import power as pow
    func = lambda x : pow(x, 4) - pow(x, 2) - 2*x + 5
    x0 = 0
    h = 0.05

    solution = advance_retreat(func, x0, h, alpha = 2).search()
    print(solution)