import autograd.numpy as np

def get_value(x):
    if type(x) == np.numpy_boxes.ArrayBox:
        return x._value
    else:
        return x


def CostOLS(target):
    
    def func(X):
        return (1.0 / target.shape[0]) * np.sum((target - X) ** 2)

    return func


def CostLogReg(target):

    def func(X):
        
        return -(1.0 / target.shape[0]) * np.sum(
            (target * np.log(X + 10e-10)) + ((1 - target) * np.log(1 - X + 10e-10))
        )

    return func


def CostCrossEntropy(target):
    
    def func(X):
        return -(1.0 / target.size) * np.sum(target * np.log(X + 10e-10))

    return func


def CostDelta(target):
    
    def func(X):
        return (1.0 / target.shape[0]) * np.sum(np.abs(target - X))

    return func

def CostOscOLS(target):
    
    def func(X):
        diff_x = X - np.roll(X,1)
        diff_x = get_value(diff_x)
        diff_x[0] = diff_x[1]
        diff_x[-1] = diff_x[-2]
        osc = np.sum((diff_x)**2)
        
        ols =  np.sum((target - X)**2)
        return (1.0 / target.shape[0])*(osc+ols)

    return func


def CostMinOls(target):

    def func(X,const):

        ols = (1.0 / target.shape[0]) * np.sum((target - X) ** 2)
        return get_value(ols) * np.sqrt(np.sum(const))

    return func
