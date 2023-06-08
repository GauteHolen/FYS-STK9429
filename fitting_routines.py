from cmath import log
from lossfunc_CFD import evolve_CFD_predict, values_to_X, XY_from_X
from hydrosolverfast import evolve_CFD_fast
import numpy as np
from nm_lib import ana_sod_shock
from schedulers import *
from copy import copy

def setup_evolve_predict(xx,rho,u,e,P,t,dt,gamma):

    def func(model_predict):
        return evolve_CFD_predict(xx,rho,u,e,P,model_predict,t,dt,gamma)
    return func


def setup_evolve_CFD_fast(xx,rho,u,e,P,t,dt,gamma):

    def func(cq,cL,cD):
        return evolve_CFD_fast(xx,rho,u,e,P,t,dt,gamma,cq=cq,cL=cL,cD=cD)
    return func



def iterate_fit_simple(model,scheduler,evolve_predict,iter,X,target, lookahead, eta = 1e-1, epochs = 50, slicing = None):

    if slicing == "linear":
        print("Linear slice")
        slice_X = slice_lookahead
    elif slicing == "log":
        print("Log slice")
        slice_X = slice_log
    else:
        print("no slice")
        slice_X = no_slice


    #Initial fit
    error = []
    err_label = []
    all_values = []
    scores = []
    consts = []
    new_X = copy(X)
    X0 = copy(X)
    X0_sliced = slice_X(X0, lookahead)
    target_sliced = slice_X(target, lookahead)

    stack = False

    for it in range(iter):
        try:
            #Run fit again with new  X and target
            #if it < iter/2:
            model.reset_weights()
            #slice
            new_X = slice_X(new_X, lookahead)
            if stack:
                new_X = np.vstack([X0_sliced,new_X.copy()])
                target = np.vstack([target_sliced,target_sliced])
            else:
                target = target_sliced

            scheduler = Momentum(eta=eta / (it+1), momentum=0.5)
            scores.append(model.fit(new_X[:],target[:], scheduler, epochs = int(epochs + epochs*it/3), batches = 1, lam = 1e-5)) # + it*epochs)

            #Get outputs from solver with model predict
            xx,t,values,const = evolve_predict(model.predict)
            consts.append(const)
            all_values.append(copy(values))
            new_X = values_to_X(values)[:-lookahead]
            
            error.append(errt(new_X,X0,lookahead))
            print("\nMean error = ", np.mean(error[-1]))
            err_label.append(f"It {it}")
            if it > 2:
                epochs += 3
        except KeyboardInterrupt:
            break
    
    return model, scores, error, err_label, all_values, consts


def errt(X,target,lookahead):
    errt = []
    for x,t in zip(X[:],target[:]):

        errt.append(np.sum((x-t)**2)/x.shape[0])
    return errt

def iterate_fit_stackX(model,scheduler,evolve_predict,evolve_fast,iter,X0,target0, target_onestep, lookahead, epochs = 50, init_const = [0.5,2,3], slicing = "linear"):
    #Reducing the X and T size:
    if slicing == "linear":
        print("Linear slice")
        slice_X = slice_lookahead
    elif slicing == "log":
        print("Log slice")
        slice_X = slice_log
    else:
        print("no slice")
        slice_X = no_slice

    all_values = []

    X = slice_X(X0,lookahead)
    target = slice_X(target0,lookahead)
    
    print("\n Coonstant cq, cL, cD")
    error = []
    err_labels = []
    scores = []
    #X and target from constant prediction
    [cq,cL,cD] = init_const
    xx,t,values,const = evolve_fast(cq,cL,cD)
    all_values.append(copy(values))
    X_const_pred = values_to_X(values)[:-(1+lookahead)]
    X_const_pred = slice_X(X_const_pred,lookahead)
    target_const_pred = slice_X(target0[1:], lookahead)

    error.append(errt(values_to_X(values),target_onestep,lookahead))
    err_labels.append("Constant c")
    print("Mean error = ", np.mean(error[-1]))
    

    #Initial fit
    if True:
        print("\n First fit")
        scores.append(model.fit(X,target,scheduler, epochs = int(epochs)))
        
        #X and target from initial shitty prediction
        xx,t,values,const = evolve_predict(model.predict)
        all_values.append(copy(values))
        X_bad_pred = values_to_X(values)[:-(1+lookahead)]
        X_bad_pred = slice_X(X_bad_pred,lookahead)
        target_bad_pred = slice_X(target0[1:], lookahead)

        error.append(errt(values_to_X(values),target_onestep,lookahead))
        err_labels.append("First bad fit")
        print("Mean error = ", np.mean(error[-1]))

        #Run first it
        print("\n First feedback fit")
        #scheduler = Momentum(eta=0.5e-2, momentum=1e-2)
        #model.reset_weights()
        scores.append(model.fit(X_bad_pred,target_bad_pred, scheduler, epochs = epochs))


    #X and target for second fit
    xx,t,values,const = evolve_predict(model.predict)
    all_values.append(copy(values))
    X_first_it = values_to_X(values)[:-(1+lookahead)]
    X_first_it = slice_X(X_first_it,lookahead)
    target_first_it = slice_X(target0[1:], lookahead)

    
    error.append(errt(values_to_X(values),target_onestep,lookahead))
    err_labels.append("First feedback fit")
    print("Mean error = ", np.mean(error[-1]))

    #Stacking all the variations
    X_stack = np.vstack([X_first_it])
    target_stack = np.vstack([target_first_it])

    print("\nRunning first stack fit")
    model.reset_weights()
    scores.append(model.fit(X_stack,target_stack, scheduler, epochs = epochs))

    xx,t,values,const = evolve_predict(model.predict)
    all_values.append(copy(values))
    X_pred_stack = values_to_X(values)

    X_pred_stack = slice_X(X_pred_stack[:-(1+lookahead)],lookahead)    
    target_pred_stack2 = slice_X(target0[1:], lookahead)

    error.append(errt(values_to_X(values),target_onestep,lookahead))
    err_labels.append("First stack fit")
    print("Mean error = ", np.mean(error[-1]))

    if True:


        X_stack2 = np.vstack([X_pred_stack])
        target_stack2 = np.vstack([target_pred_stack2])

        print("\nRunning stack fit feedback")
        model.reset_weights()
        scores.append(model.fit(X_stack2,target_stack2, scheduler, epochs = epochs*2))
        
        xx,t,values,const = evolve_predict(model.predict)
        all_values.append(copy(values))
        X_pred_stack2 = values_to_X(values)[:-(1+lookahead)]
        X_pred_stack2 = slice_X(X_pred_stack2,lookahead)
        target_pred_stack2 = slice_X(target0[1:], lookahead)


        error.append(errt(values_to_X(values),target_onestep,lookahead))
        err_labels.append("Final stack fit")
        print("Mean error = ", np.mean(error[-1]))




    
    return model, scores, error, err_labels, all_values



def slice_lookahead(X,lookahead):
    step = int(lookahead/5)
    return X[::step,:]

def slice_log(X,lookahead):
    Nx = X.shape[0]
    N = min(lookahead,5)*int((Nx-1)/lookahead)
    #print(N,Nx)

    #idx = np.hstack([[0],np.geomspace(1,Nx-1,num=N)])
    idx = np.geomspace(1e-3,Nx-1,num=N)

    X_log = X[idx.astype(int),:]
    return X_log

def no_slice(X, lookahead):
    return X

