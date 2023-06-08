"""Modified ffnn for fitting constants for numerical methods

"""


import math
import autograd.numpy as np
import sys
import warnings
from autograd import grad, elementwise_grad
from random import random, seed
from copy import deepcopy, copy
from typing import Tuple, Callable
from sklearn.utils import resample
from hydrosolverfast import predictTimestepJit

from costfunc import *
from schedulers import *
from lossfunc_CFD import CostDiff, evolve_CFD_predict

warnings.simplefilter("error")

def get_value(x):
    if type(x) == np.numpy_boxes.ArrayBox:
        return x._value
    else:
        return x

def identity(X):
    return X


def sigmoid(X):
    try:
        return 1.0 / (1 + np.exp(-X))
    except FloatingPointError:
        return np.where(X > np.zeros(X.shape), np.ones(X.shape), np.zeros(X.shape))


def tanh(X):
    return np.tanh(X)

def absolute(X):
    return np.abs(X)

def softmax(X):
    X = X - np.max(X, axis=-1, keepdims=True)
    delta = 10e-10
    return np.exp(X) / (np.sum(np.exp(X), axis=-1, keepdims=True) + delta)


def RELU(X):
    return np.where(X > np.zeros(X.shape), X, np.zeros(X.shape))


def LRELU(X):
    delta = 10e-4
    return np.where(X > np.zeros(X.shape), X, delta * X)


def derivate(func):
    if func.__name__ == "RELU":

        def func(X):
            return np.where(X > 0, 1, 0)

        return func

    elif func.__name__ == "LRELU":

        def func(X):
            delta = 10e-4
            return np.where(X > 0, 1, delta)

        return func

    elif func.__name__ == "absolute":

        return elementwise_grad(func)

    else:
        return elementwise_grad(func)

class FFNN:
    """
    Description:
    ------------
        Feed Forward Neural Network with interface enabling flexible design of a
        nerual networks architecture and the specification of activation function
        in the hidden layers and output layer respectively. This model can be used
        for both regression and classification problems, depending on the output function.

    Attributes:
    ------------
        I   dimensions (tuple[int]): A list of positive integers, which specifies the
            number of nodes in each of the networks layers. The first integer in the array
            defines the number of nodes in the input layer, the second integer defines number
            of nodes in the first hidden layer and so on until the last number, which
            specifies the number of nodes in the output layer.
        II  hidden_func (Callable): The activation function for the hidden layers
        III output_func (Callable): The activation function for the output layer
        IV  cost_func (Callable): Our cost function
        V   seed (int): Sets random seed, makes results reproducible
    """

    def __init__(
        self,
        dimensions: tuple[int],
        hidden_func: Callable = sigmoid,
        output_func: Callable = absolute,
        cost_func: Callable = CostOLS,
        seed: int = None,
    ):
        self.dimensions = dimensions
        self.hidden_func = hidden_func
        self.output_func = output_func
        self.cost_func = cost_func
        self.seed = seed
        self.weights = list()
        self.schedulers_weight = list()
        self.schedulers_bias = list()
        self.a_matrices = list()
        self.z_matrices = list()
        self.classification = None

        self.reset_weights()
        self._set_classification()

    def fit(
        self,
        X: np.ndarray,
        t: np.ndarray,
        scheduler: Scheduler,
        batches: int = 1,
        epochs: int = 100,
        lam: float = 0,
        X_val: np.ndarray = None,
        t_val: np.ndarray = None,

    ):
        """
        Description:
        ------------
            This function performs the training the neural network by performing the feedforward and backpropagation
            algorithm to update the networks weights.

        Parameters:
        ------------
            I    X (np.ndarray) : training data
            II   t (np.ndarray) : target data
            III  scheduler (Scheduler) : specified scheduler (algorithm for optimization of gradient descent)
            IV   scheduler_args (list[int]) : list of all arguments necessary for scheduler

        Optional Parameters:
        ------------
            V    batches (int) : number of batches the datasets are split into, default equal to 1
            VI   epochs (int) : number of iterations used to train the network, default equal to 100
            VII  lam (float) : regularization hyperparameter lambda
            VIII X_val (np.ndarray) : validation set
            IX   t_val (np.ndarray) : validation target set

        Returns:
        ------------
            I   scores (dict) : A dictionary containing the performance metrics of the model.
                The number of the metrics depends on the parameters passed to the fit-function.

        """

        # setup 
        if self.seed is not None:
            np.random.seed(self.seed)

        val_set = False
        if X_val is not None and t_val is not None:
            val_set = True

        # creating arrays for score metrics
        train_errors = np.empty(epochs)
        train_errors.fill(np.nan)
        val_errors = np.empty(epochs)
        val_errors.fill(np.nan)

        train_accs = np.empty(epochs)
        train_accs.fill(np.nan)
        val_accs = np.empty(epochs)
        val_accs.fill(np.nan)

        self.schedulers_weight = list()
        self.schedulers_bias = list()

        batch_size = X.shape[0] // batches

        X, t = resample(X, t)

        # this function returns a function valued only at X
        cost_function_train = self.cost_func(t)
        if val_set:
            cost_function_val = self.cost_func(t_val)

        # create schedulers for each weight matrix
        for i in range(len(self.weights)):
            self.schedulers_weight.append(copy(scheduler))
            self.schedulers_bias.append(copy(scheduler))

        print(f"{scheduler.__class__.__name__}: Eta={scheduler.eta}, Lambda={lam}")

        try:
            for e in range(epochs):
                for i in range(batches):
                    # allows for minibatch gradient descent
                    if i == batches - 1:
                        # If the for loop has reached the last batch, take all thats left
                        X_batch = X[i * batch_size :, :]
                        t_batch = t[i * batch_size :, :]
                    else:
                        X_batch = X[i * batch_size : (i + 1) * batch_size, :]
                        t_batch = t[i * batch_size : (i + 1) * batch_size, :]

                    self._feedforward(X_batch)
                    self._backpropagate(X_batch, t_batch, lam)
    

                # reset schedulers for each epoch (some schedulers pass in this call)
                for scheduler in self.schedulers_weight:
                    scheduler.reset()

                for scheduler in self.schedulers_bias:
                    scheduler.reset()

                # computing performance metrics
                #pred_train = self.predict_timestep(X)
                #train_error = cost_function_train(pred_train)
                #errt = np.zeros(X.shape[0])
                #for x,y,j in zip(pred_train, t, range(X.shape[0])):
                #    errt[i] = np.sum((x-y)**2)
                #train_error = np.mean(errt)
                if e/10 == int(e/10):
                    train_error = self.error_evolve_cfd_predict()

                train_errors[e] = train_error
                if val_set:
                    
                    pred_val = self.predict_timestep(X_val)
                    val_error = cost_function_val(pred_val)
                    val_errors[e] = val_error

                if self.classification:
                    train_acc = self._accuracy(self.predict_timestep(X), t)
                    train_accs[e] = train_acc
                    if val_set:
                        val_acc = self._accuracy(pred_val, t_val)
                        val_accs[e] = val_acc

                # printing progress bar
                progression = e / epochs
                print_length = self._progress_bar(
                    progression,
                    train_error=train_errors[e],
                    train_acc=train_accs[e],
                    val_error=val_errors[e],
                    val_acc=val_accs[e],
                )

                
        except KeyboardInterrupt:
            # allows for stopping training at any point and seeing the result
            pass

        # visualization of training progression (similiar to tensorflow progression bar)
        sys.stdout.write("\r" + " " * print_length)
        sys.stdout.flush()
        self._progress_bar(
            1,
            train_error=train_errors[e],
            train_acc=train_accs[e],
            val_error=val_errors[e],
            val_acc=val_accs[e],
        )
        sys.stdout.write("")

        # return performance metrics for the entire run
        scores = dict()

        scores["train_errors"] = train_errors

        if val_set:
            scores["val_errors"] = val_errors

        if self.classification:
            scores["train_accs"] = train_accs

            if val_set:
                scores["val_accs"] = val_accs

        return scores

    def predict_timestep(self,X):
        """For estimating error, does update prediction

        Args:
            X (_type_): _description_

        Returns:
            _type_: _description_
        """

        

        pred_y = X
        for _ in range(self.Nt):
            const_pred = self.predict(pred_y)
            pred_y = predictTimestepJit(pred_y.copy(),const_pred,self.dx,self.dt,self.gamma)
    
        return pred_y

    def predict_one_timestep(self,X):
        const_pred = self.predict(X)
        return predictTimestepJit(X,const_pred,self.dx,self.dt,self.gamma)

    def timestep_from_predict(self,const_pred,X):
        """For setting weights, does not update prediction

        Args:
            const_pred (_type_): _description_
            X (_type_): _description_

        Returns:
            _ttype_: _description_
        """
        pred_y = X.copy()
        const_pred = self.predict(pred_y)
        for _ in range(self.Nt):
            pred_y = predictTimestepJit(pred_y.copy(),const_pred,self.dx,self.dt,self.gamma)
            #const_pred = self.predict(X_new)
        return pred_y

    

    def predict(self, X: np.ndarray, *, threshold=0.5):
        """
         Description:
         ------------
             Performs prediction after training of the network has been finished.

         Parameters:
        ------------
             I   X (np.ndarray): The design matrix, with n rows of p features each

         Optional Parameters:
         ------------
             II  threshold (float) : sets minimal value for a prediction to be predicted as the positive class
                 in classification problems

         Returns:
         ------------
             I   z (np.ndarray): A prediction vector (row) for each row in our design matrix
                 This vector is thresholded if regression=False, meaning that classification results
                 in a vector of 1s and 0s, while regressions in an array of decimal numbers

        """

        predict = self._feedforward(X)

        if self.classification:
            return np.where(predict > threshold, 1, 0)
        else:
            return predict


    def set_CFD_params(self,xx,dt,gamma, Nt = 1):
        self.xx = xx
        self.dx = (xx[1]-xx[0])*np.ones(xx.shape)
        self.dt = dt
        self.gamma = gamma
        self.Nt = Nt

    def setup_evolve_predict(self, analytic_values, evolve_t):
        
        ut,rhot,Pt,et = analytic_values[1:]
        self.evovle_t = evolve_t

        # storing analytical values
        self.rhot = rhot
        self.ut = ut
        self.et = et
        self.Pgt = Pt

    def error_evolve_cfd_predict(self):
        
        xx,t,values,const = evolve_CFD_predict(self.xx,
                                                self.rhot[0],self.ut[0],self.et[0],self.Pgt[0],
                                                self.predict,self.evovle_t, self.dt,self.gamma)

        ut,rhot,Pt,et = values[1:]
        err_rho = 0
        err_u = 0
        err_e = 0
        err_P = 0
        Nt = ut.shape[0]
        Nx = ut.shape[1]

        #for i in range(Nt-1):
        err_rho += np.sum((rhot[:]-self.rhot[:])**2)
        err_u += np.sum((ut[:]-self.ut[:])**2)
        err_e += np.sum((et[:]-self.et[:])**2)
        err_P += np.sum((Pt[:]-self.Pgt[:])**2)
        
        err_rho = err_rho/(Nx*Nt)
        err_u = err_u/(Nx*Nt)
        err_e = err_e/(Nx*Nt)
        err_P = err_P/(Nx*Nt)

        #print(f"\nEvolve predict vs analytical solutions for {Nt} timesteps:")
        #print(f"rho = {err_rho}\t u = {err_u}\t e = {err_e}\t Pg = {err_P}")

        return (err_rho+err_e+err_u+err_P)/4


    def reset_weights(self):
        """
        Description:
        ------------
            Resets/Reinitializes the weights in order to train the network for a new problem.

        """
        if self.seed is not None:
            np.random.seed(self.seed)

        self.weights = list()
        for i in range(len(self.dimensions) - 1):
            weight_array = np.random.randn(
                self.dimensions[i] + 1, self.dimensions[i + 1]
            )
            weight_array[0, :] = np.random.randn(self.dimensions[i + 1]) * 0.01

            self.weights.append(weight_array)

    def _feedforward(self, X: np.ndarray):
        """
        Description:
        ------------
            Calculates the activation of each layer starting at the input and ending at the output.
            Each following activation is calculated from a weighted sum of each of the preceeding
            activations (except in the case of the input layer).

        Parameters:
        ------------
            I   X (np.ndarray): The design matrix, with n rows of p features each

        Returns:
        ------------
            I   z (np.ndarray): A prediction vector (row) for each row in our design matrix
        """

        # reset matrices
        self.a_matrices = list()
        self.z_matrices = list()

        # if X is just a vector, make it into a matrix
        if len(X.shape) == 1:
            X = X.reshape((1, X.shape[0]))

        # Add a coloumn of zeros as the first coloumn of the design matrix, in order
        # to add bias to our data
        bias = np.ones((X.shape[0], 1)) * 0.01
        X = np.hstack([bias, X])

        # a^0, the nodes in the input layer (one a^0 for each row in X - where the
        # exponent indicates layer number).
        a = X
        self.a_matrices.append(a)
        self.z_matrices.append(a)

        # The feed forward algorithm
        for i in range(len(self.weights)):
            if i < len(self.weights) - 1:
                z = a @ self.weights[i]
                self.z_matrices.append(z)
                a = self.hidden_func(z)
                # bias column again added to the data here
                bias = np.ones((a.shape[0], 1)) * 0.01
                a = np.hstack([bias, a])
                self.a_matrices.append(a)
            else:
                try:
                    # a^L, the nodes in our output layers
                    z = a @ self.weights[i]
                    a = self.output_func(z)
                    self.a_matrices.append(a)
                    self.z_matrices.append(z)
                except Exception as OverflowError:
                    print(
                        "OverflowError in fit() in FFNN\nHOW TO DEBUG ERROR: Consider lowering your learning rate or scheduler specific parameters such as momentum, or check if your input values need scaling"
                    )

        # this will be a^L
        #print(a)
        return a


    def reshape_delta_mat(self,a_mat,delta_matrix_Nt):
        delta_matrix = np.zeros(a_mat.shape)
        N_out = delta_matrix.shape[1]
        Ni = delta_matrix.shape[1]
        Nti = delta_matrix_Nt.shape[1]
        #print(Ni, Nti)

        size = int(Nti/Ni)
        for j in range(delta_matrix.shape[0]):
            combined = np.zeros(Ni)
            rho,u,e,Pg = np.split(delta_matrix_Nt[j], 4)
            
            combined[0] = np.mean(e+Pg)
            combined[1] = np.mean(u)
            combined[2] = np.mean(rho)

            delta_matrix[j] = combined
        return delta_matrix


    def _backpropagate(self, X: np.ndarray, t: np.ndarray, lam: float):
        """
        Description:
        ------------
            Performs the backpropagation algorithm. In other words, this method
            calculates the gradient of all the layers starting at the
            output layer, and moving from right to left accumulates the gradient until
            the input layer is reached. Each layers respective weights are updated while
            the algorithm propagates backwards from the output layer (auto-differentation in reverse mode).

        Parameters:
        ------------
            I   X (np.ndarray): The design matrix, with n rows of p features each.
            II  t (np.ndarray): The target vector, with n rows of p targets.
            III lam (float32): regularization parameter used to punish the weights in case of overfitting

        Returns:
        ------------
            No return value.

        """
        out_derivative = derivate(self.output_func)
        hidden_derivative = derivate(self.hidden_func)
        #print(len(self.a_matrices))
        #for amat in self.a_matrices:
        #    print(amat.shape)
        #print("len weights",len(self.weights))
        #print(np.max(np.abs(self.weights[-1])))

        for i in range(len(self.weights) - 1, -1, -1):
            #print("i",i)
  
            # delta terms for output
            if i == len(self.weights) - 1:
                #delta_matrix_Nt = - self.timestep_from_predict(self.a_matrices[i + 1],X) + t
                X_pred = self.timestep_from_predict(self.a_matrices[i + 1],X)
                #W_pred = self.timestep_from_predict(self.z_matrices[i + 1],X)
                #out_deriv = out_derivative(self.z_matrices[i + 1])
                cost_func_derivative = grad(self.cost_func(t))
                cost_deriv = cost_func_derivative(X_pred)
                #w_deriv = cost_func_derivative(W_pred)
                #out_deriv = out_derivative(self.z_matrices[i + 1])
                cost_deriv = self.reshape_delta_mat(self.z_matrices[i + 1], cost_deriv)
                #w_deriv = self.reshape_delta_mat(self.z_matrices[i + 1], w_deriv)
                delta_matrix = cost_deriv
                #delta_matrix = out_derivative(self.z_matrices[i + 1] * cost_deriv)
                #delta_matrix = cost_deriv * out_deriv
                #if np.min(delta_matrix) < 0:
                    #print(np.min(delta_matrix))
                    #print(delta_matrix)
                #print("max delta",np.max(np.abs(delta_matrix)))
                #print("mean delta",np.mean(delta_matrix))
                # delta_matrix_Nt =  out * cost_deriv
                
                #delta_matrix = self.reshape_delta_mat(self.a_matrices[i + 1],delta_matrix_Nt)
                #delta_matrix *= 1e2
                #print(np.max(np.abs(delta_matrix)))

                """
                # for multi-class classification
                if self.output_func.__name__ == "softmax":
                    delta_matrix_Nt = self.timestep_from_predict(self.a_matrices[i + 1],X) - t
                # for binary classification
                else:
                    cost_func_derivative = grad(self.cost_func(t))
                    #print(self.a_matrices[i + 1].shape, self.z_matrices[i + 1].shape)
                    #print(cost_func_derivative)
                    #a_mat_predy = self.timestep_from_predict(self.a_matrices[i + 1],X)
                    out_deriv = self.timestep_from_predict(out_derivative(self.z_matrices[i + 1]),X)
                    cost_deriv = cost_func_derivative(self.timestep_from_predict(self.a_matrices[i + 1],X))
                    #print(out_deriv.shape,cost_deriv.shape)
                    delta_matrix_Nt =  out_deriv * cost_deriv
                    print(np.max(np.abs(delta_matrix_Nt)))
                    print(np.max(np.abs(out_deriv)))
                    print(np.max(np.abs(cost_deriv)))
                delta_matrix = self.reshape_delta_mat(self.a_matrices[i + 1],delta_matrix_Nt)
                delta_matrix *= 1e3
                """


            # delta terms for hidden layer
            else:
                delta_matrix = (
                    self.weights[i + 1][1:, :] @ delta_matrix.T
                ).T * hidden_derivative(self.z_matrices[i + 1])
            #print(delta_matrix)
            # calculate gradient
            gradient_weights = self.a_matrices[i][:, 1:].T @ delta_matrix
            gradient_bias = np.sum(delta_matrix, axis=0).reshape(
                1, delta_matrix.shape[1]
            )

            # regularization term
            gradient_weights += self.weights[i][1:, :] * lam

            # use scheduler
            update_matrix = np.vstack(
                [
                    self.schedulers_bias[i].update_change(gradient_bias),
                    self.schedulers_weight[i].update_change(gradient_weights),
                ]
            )

            # update weights and bias
            self.weights[i] -= update_matrix

    def _accuracy(self, prediction: np.ndarray, target: np.ndarray):
        """
        Description:
        ------------
            Calculates accuracy of given prediction to target

        Parameters:
        ------------
            I   prediction (np.ndarray): vector of predicitons output network
                (1s and 0s in case of classification, and real numbers in case of regression)
            II  target (np.ndarray): vector of true values (What the network ideally should predict)

        Returns:
        ------------
            A floating point number representing the percentage of correctly classified instances.
        """
        assert prediction.size == target.size
        return np.average((target == prediction))
    
    def _set_classification(self):
        """
        Description:
        ------------
            Decides if FFNN acts as classifier (True) og regressor (False),
            sets self.classification during init()
        """
        self.classification = False
        if (
            self.cost_func.__name__ == "CostLogReg"
            or self.cost_func.__name__ == "CostCrossEntropy"
        ):
            self.classification = True

    def _progress_bar(self, progression, **kwargs):
        """
        Description:
        ------------
            Displays progress of training
        """
        print_length = 40
        num_equals = int(progression * print_length)
        num_not = print_length - num_equals
        arrow = ">" if num_equals > 0 else ""
        bar = "[" + "=" * (num_equals - 1) + arrow + "-" * num_not + "]"
        perc_print = self._format(progression * 100, decimals=5)
        line = f"  {bar} {perc_print}% "

        for key in kwargs:
            if not np.isnan(kwargs[key]):
                value = self._format(kwargs[key], decimals=4)
                line += f"| {key}: {value} "
        sys.stdout.write("\r" + line)
        sys.stdout.flush()
        return len(line)

    def _format(self, value, decimals=4):
        """
        Description:
        ------------
            Formats decimal numbers for progress bar
        """
        if value > 0:
            v = value
        elif value < 0:
            v = -10 * value
        else:
            v = 1
        n = 1 + math.floor(math.log10(v))
        if n >= decimals - 1:
            return str(round(value))
        return f"{value:.{decimals-n-1}f}"


