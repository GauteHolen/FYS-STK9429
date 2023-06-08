import numpy as np
from numpy import copy as copy
from numba import njit,prange

@njit()
def ddx(dx, hh):
    r"""
    returns the centered 2nd derivative of hh respect to xx. 

    Parameters
    ---------- 
    dx : `array`
        delta x. 
    hh : `array`
        Function that depends on xx. 

    Returns
    -------
    `array`
        The centered 2nd order derivative of hh respect to xx. First 
        and last grid points are ill calculated. 
    """
    for i in range(len(hh)):
        hh[i] = 0.5 * hh[i] / dx[i]
    dhdx = np.zeros(len(hh))
    for i in range(1,len(hh)-1):
        dhdx[i] = hh[i+1]-hh[i-1]

    dhdx[0] = dhdx[2]
    dhdx[-1] = dhdx[-3]

    return dhdx


@njit()
def ddx1(dx, hh):
    r"""
    returns the centered 2nd derivative of hh respect to xx. 

    Parameters
    ---------- 
    dx : `array`
        delta x. 
    hh : `array`
        Function that depends on xx. 

    Returns
    -------
    `array`
        The centered 2nd order derivative of hh respect to xx. First 
        and last grid points are ill calculated. 
    """
    for i in range(len(hh)):
        hh[i] = hh[i] / (dx[i]*12)
    dhdx = np.zeros(len(hh))
    for i in range(2,len(hh)-2):
        dhdx[i] = hh[i-2]-8*hh[i-1] + 8*hh[i+1] - hh[i+2]

    dhdx[0] = dhdx[3]
    dhdx[1] = dhdx[3]
    dhdx[2] = dhdx[3]
    dhdx[-1] = dhdx[-4]
    dhdx[-2] = dhdx[-4]
    dhdx[-3] = dhdx[-4]

    return dhdx


@njit()
def step_pressure(e,rho,u,gamma):
    N = len(u)
    P_new = np.zeros(N, dtype = np.float64)
    for i in range(N):
        P_new[i] = ((gamma-1)*rho[i])*(e[i]/rho[i]-u[i]*u[i])
        #P_new[i] = ((gamma-1))*(e[i] - rho[i]-u[i]*u[i])
    return P_new


@njit()
def step_momentum(dx,u,rho,Pg,dt,cq = 0,cL = 0):
    N = len(u)
    q = get_q_diffusive(dx,copy(rho),copy(u),cq, cL)
    mom = np.zeros(N, dtype = np.float64)
    rhouu = np.zeros(N, dtype = np.float64)
    Pgq = np.zeros(N, dtype = np.float64)
    for i in range(N):
        rhouu[i] = rho[i] * u[i] * u[i]
        Pgq[i] = Pg[i] + q[i]
    #print(rhouu)
    #print("rhouu",rhouu)
    ddx1 = ddx(dx,rhouu)
    #print("ddx1",ddx1)
    ddx2 = ddx(dx,Pgq)

    for i in range(N):
        mom[i] = u[i]*rho[i] - dt*(ddx1[i] + ddx2[i])

    return mom 


@njit()
def step_density(dx,u,rho,dt, cD = 0):
    """_summary_

    Args:
        xx (_type_): _description_
        u (_type_): _description_
        rho (_type_): _description_
        dt (_type_): _description_
        y (deriv_cent): _description_
        ddx (_type_, optional): _description_. Defaults to lambdax.

    Returns:
        _type_: _description
    """

    #art_diff = dx**2 * step_diff_burgers(xx,rho, a=u)
    N = len(u)
    rhou = np.zeros(N, dtype = np.float64)
    for i in range(N):
        rhou[i] = rho[i]*u[i]

    deriv1 = ddx(dx,rhou)
    art_diff = np.zeros(N, dtype = np.float64)
    for i in range(N):
        art_diff[i] = - cD*rho[i]*dx[i]*deriv1[i]
    

    #rho_new = rho - dt*ddx(xx,rho*u) + dt*ddx(xx,Diff*ddx(xx,rho*u))
    #rho_new = rho - dt*ddx(xx,rho*u)
    rhoartdiffu = np.zeros(N, dtype = np.float64)
    for i in range(N):
        rhoartdiffu[i] = (rho[i]+art_diff[i])*u[i]
    deriv2 = ddx(dx,rhoartdiffu)
    rho_new = np.zeros(N, dtype = np.float64)
    for i in range(N):
        rho_new[i] = rho[i] - dt*deriv2[i]

    return rho_new

@njit()
def step_energy(dx,e,u,Pg,rho,dt, cq = 0, cL = 0):
    """_summary_

    Args:
        xx (_type_): _description_
        e (_type_): _description_
        u (_type_): _description_
        Pg (_type_): _description_
        dt (_type_): _description_
        y (deriv_cent): _description_
        ddx (_type_, optional): _description_. Defaults to lambdax.

    Returns:
        _type_: _description_
    """
    #print(np.amax(np.abs(Q_col)))
    N = len(u)
    q = get_q_diffusive(dx,copy(rho),copy(u),cq,cL)
    #q = np.zeros(N)
    #print("Q_col",Q_col[0])
    eu = np.zeros(N, dtype = np.float64)
    for i in range(N):
        eu[i] = e[i] * u[i]
    deriv1 = ddx(dx,eu)
    deriv2 = ddx(dx,u)
    e_new = np.zeros(N, dtype = np.float64)
    for i in range(N):
        e_new[i] = e[i] - dt*deriv1[i] - dt*(Pg[i]+q[i])*deriv2[i]

    return e_new

@njit()
def step_euler_diffusive_fast(rho,u,e,Pg,gamma,dx,dt,cq, cL, cD):
    #u = np.pad(u,bnd_limits,bnd_type)
    #rho = np.pad(rho,bnd_limits,bnd_type)
    #e = np.pad(e,bnd_limits,bnd_type)
    #Pg = np.pad(Pg,bnd_limits,bnd_type)
    #xx = np.pad(xx,bnd_limits,"reflect", reflect_type='odd')

    mom = step_momentum(copy(dx),copy(u),copy(rho),copy(Pg),dt,cq = cq, cL = cL)
    #print("mom",mom)
    u_new = np.zeros(len(mom))
    for i in range(len(mom)):
        u = mom[i] / rho[i]
        c = np.sqrt(gamma*Pg[i]/rho[i])
        u_new[i] = min(u,c)
    #print("u_new1",u_new)
    e_new = step_energy(copy(dx),copy(e),copy(u_new),copy(Pg),copy(rho),dt,cq=cq,cL=cL)
    #print("u_new2",u_new)
    rho_new = step_density(copy(dx),copy(u_new),copy(rho),dt,cD=cD)
    #print("u_new3",u_new)
    Pg_new = step_pressure(copy(e_new),copy(rho_new),copy(u_new),gamma)
    #[rho_n,u_n,e_n,Pg_n] = unpad([rho_new,u_new,e_new,Pg_new],bnd_limits)
    #print("u_new4",u_new)
    return rho_new,u_new,e_new,Pg_new




@njit()
def get_q_diffusive(dx,rho,u,cq, cL):

    deriv1 = ddx(copy(dx),copy(u))
    deriv2 = ddx(copy(dx),copy(deriv1))
    N = len(u)

    qdiff = np.zeros(N, dtype = np.float64)
    for i in range(N):
        q_RvN = cq*dx[i]*dx[i]*deriv2[i]
        qLQ = cL*dx[i]*deriv1[i]
        qdiff[i] = - rho[i] * (q_RvN + qLQ)

    return qdiff

@njit
def predictTimestepJit(X,const_pred,dx,dt,gamma):
    Nx = X.shape[1]
    Nu = int(Nx/4)
    X_pred = np.zeros(X.shape)
    
    
    for i in prange(X.shape[0]):
        [cq,cL,cD] = const_pred[i]
        #print(cq,cL,cD)
        x = X[i]
        rho,u,e,Pg = x.reshape(4,int(x.shape[-1]/4))

        #print("here3")
        X_combine = np.zeros(Nx)
        rho_new,u_new,e_new,Pg_new = step_euler_diffusive_fast(copy(rho),copy(u),copy(e),copy(Pg),gamma,dx,dt,cq,cL,cD)
        #print("here4")
        #X_combine = np.array([rho_new,u_new,e_new,Pg_new])
        
        #cost += np.sum( (y-np.ravel(X_combine))**2)
        #if i == 1:
        #    print("rhp_new",rho_new)
        
        for j in range(Nu):
            X_combine[j] = rho_new[j]
            X_combine[Nu+j] = u_new[j]
            X_combine[2*Nu+j] = e_new[j]
            X_combine[3*Nu+j] = Pg_new[j]

        X_pred[i] = copy(X_combine)
    return X_pred

@njit
def FastCostDiffJit(X,y,const_pred,dx,dt,gamma):
    cost = 0
    Nx = X.shape[1]
    Nu = int(Nx/4)
    new_X = np.zeros(X.shape)
    
    
    for i in prange(X.shape[0]):
        [cq,cL,cD] = const_pred[i]
        #print(cq,cL,cD)
        x = X[i]
        rho,u,e,Pg = x.reshape(4,int(x.shape[-1]/4))

        #print("here3")
        X_combine = np.zeros(Nx)
        rho_new,u_new,e_new,Pg_new = step_euler_diffusive_fast(copy(rho),copy(u),copy(e),copy(Pg),gamma,dx,dt,cq,cL,cD)
        #print("here4")
        #X_combine = np.array([rho_new,u_new,e_new,Pg_new])
        
        #cost += np.sum( (y-np.ravel(X_combine))**2)
        #if i == 1:
        #    print("rhp_new",rho_new)
        
        for j in range(Nu):
            X_combine[j] = rho_new[j]
            X_combine[Nu+j] = u_new[j]
            X_combine[2*Nu+j] = e_new[j]
            X_combine[3*Nu+j] = Pg_new[j]

        new_X[i] = copy(X_combine)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            cost += (y[i][j] - new_X[i][j])**2

    return cost / X.shape[0]


def evolve_CFD_fast(xx,rho,u,e,Pg,t,dt,gamma, cq = 0.5,cL = 2,cD  = 3):
    e = Pg/((gamma-1))+ 0.5*rho*u**2
    rhot = np.zeros((t.shape[0],xx.shape[0]))
    ut = np.zeros((t.shape[0],xx.shape[0]))
    et = np.zeros((t.shape[0],xx.shape[0]))
    Pgt = np.zeros((t.shape[0],xx.shape[0]))
    const = np.zeros((t.shape[0],3))
    rhot[0],ut[0],et[0],Pgt[0] = rho,u,e,Pg
    dx = np.ones(xx.shape[0])*(xx[1]-xx[0])
    Nt = len(t)
    for i in range(1,Nt):
        #print(f"Timestep {i} of {Nt}", end = "\r")
        X = np.ravel(np.array([rho,u,e,Pg]))
        const[i] = [cq,cL,cD]


        rho_new,u_new,e_new,Pg_new = step_euler_diffusive_fast(rho,u,e,Pg,gamma,dx,dt,cq, cL, cD)
        #print("u_new elolve",u_new)
        rhot[i],ut[i],et[i],Pgt[i] = rho_new,u_new,e_new,Pg_new
        rho,u,e,Pg = rho_new,u_new,e_new,Pg_new

    values = ["jit test", ut,rhot,Pgt,et]
    return xx,t,values,const