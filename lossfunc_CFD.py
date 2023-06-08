import autograd.numpy as np
import nm_lib as nm
from costfunc import CostOLS
from hydrosolverfast import step_euler_diffusive_fast, FastCostDiffJit
from numba import njit,prange

def get_value(x):
    if type(x) == np.numpy_boxes.ArrayBox:
        #print(f"arraybox converted")
        return x._value
    else:
        #print(type(x))
        return x

def values_to_X(values):
    ut,rhot,Pt,et = values[1:]
    #print(ut.shape)

    X = np.ndarray((ut.shape[0],int(ut.shape[1]*4)))
    for i in range(X.shape[0]):
        X[i] = np.ravel(np.array([rhot[i],ut[i],et[i],Pt[i]]))
    
    return X

def XY_from_X(X, Nt = 1):
    y = X[Nt:,:]
    return X[:-Nt,:],y,y[1:-Nt+1,:]



def CostDiff(y):

    y = get_value(y)
    #@njit
    def func(const_diff,X,dx,dt,gamma):
        #X = get_value(X)
        #print("cost_func const diff:", const_diff)
        const = get_value(const_diff)

        cost = FastCostDiffJit(X,y,const,dx,dt,gamma)
        print("cost = ", cost)
        return cost
        
    return func
    

def evolve_CFD_predict(xx,rho,u,e,Pg,predict_c,t,dt,gamma):
    e = Pg/((gamma-1))+ 0.5*rho*u**2
    rhot = np.zeros((t.shape[0],xx.shape[0]))
    ut = np.zeros((t.shape[0],xx.shape[0]))
    et = np.zeros((t.shape[0],xx.shape[0]))
    Pgt = np.zeros((t.shape[0],xx.shape[0]))
    const = np.zeros((t.shape[0],3))
    rhot[0],ut[0],et[0],Pgt[0] = rho,u,e,Pg
    dx = (xx[1]-xx[0])*np.ones(len(xx))

    for i in range(1,len(t)):
        X = np.ravel(np.array([rho,u,e,Pg]))
        [cq,cL,cD]  = predict_c(X)[0]
        const[i] = [cq,cL,cD]

        
        rho_new,u_new,e_new,Pg_new = step_euler_diffusive_fast(rho,u,e,Pg,gamma,dx,dt,cq, cL, cD)
        rhot[i],ut[i],et[i],Pgt[i] = rho_new,u_new,e_new,Pg_new
        rho,u,e,Pg = rho_new,u_new,e_new,Pg_new

    values = ["CFD predict", ut,rhot,Pgt,et]
    return xx,t,values,const


if __name__ == "__main__":

    nump = 10
    cfl_cut = 0.1
    Nt = 10
    dt = 0.01
    gamma = 5/3
    t = np.linspace(0,Nt*dt,Nt+1)
    print(t)

    xx,u,rho,Pg = nm.init_sod_test(nump,1.0,0.125,1.0/gamma,0.125/gamma)

    values = nm.ana_sod_shock(xx,gamma,t, Pg[0],Pg[-1],rho[0],rho[-1])

    X = values_to_X(values)
    print(X[0])
    X,y = XY_from_X(X)
    print(X.shape,y.shape)

    costfunc = CostDiff(y[0])
    costfunc(X[0])



