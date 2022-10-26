# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 16:34:41 2022

@author: Chakron.D
"""
# %% Importing the libraries

# numpy
import numpy as np
import copy

# PyTroch
import torch
import torch.nn as nn
import torch.nn.functional as F

# %% AE Class (Auto Encoder)


class AE_Class(nn.Module):

    # Inheritance from parents class
    def __init__(self):
        super().__init__()
    
    
    def setParams(self,
                 feature_in,
                 feature_out,
                 n_enUnit,
                 n_latUnit,
                 learning_rate=0.01,
                 activation_fun='relu',  # activation function at hidden layers
                 activation_fun_out='sigmoid',
                 optim_fun='Adam',
                 w_fro=''
                 ):


        # ---------------------------------------
        # Layer structure
        # ---------------------------------------
        self.layers = nn.ModuleDict()  # store layers

        self.layers['input']   = nn.Linear(feature_in, n_enUnit)  # input layer

        self.layers['encoder'] = nn.Linear(n_enUnit, n_latUnit)
        
        self.layers['latent']  = nn.Linear(n_latUnit, n_enUnit)

        self.layers['decoder'] = nn.Linear(n_enUnit, feature_out)  # decoder layer

        # ---------------------------------------
        # Parameters
        # ---------------------------------------
        # Learning rate
        self.lr = learning_rate
        # activaiton funciton
        self.actfun = activation_fun
        self.actfunOut = activation_fun_out
        # weight params
        self.wFro = w_fro

        # ---------------------------------------
        # Number of parameters
        # ---------------------------------------
        # count the total number of weights in the model weight except bias term
        n_weights = 0
        for param_name, weight in self.named_parameters():
            if 'bias' not in param_name:
                n_weights = n_weights + weight.numel()

        self.nWeights = n_weights

        # ---------------------------------------
        # Optimizer
        # ---------------------------------------
        optimFun = getattr(torch.optim, optim_fun)
        
        # parameters dict
        paramsDict = {
          'lr': learning_rate,
        }
        
        self.optimizer = optimFun(self.parameters(), **paramsDict)
        

    # forward pass
    def forward(self, x):

        # activation functions
        actFun = getattr(torch, self.actfun)
        # activation functions decoder
        actFun_out = getattr(torch, self.actfunOut)

        # input layer
        x = actFun(self.layers['input'](x))

        # encoder layer
        x = actFun(self.layers['encoder'](x))
        self.fwLat = x.clone().detach() # observe this output
        
        # laten layer
        x = actFun(self.layers['latent'](x))

        # return decoder layer
        x = actFun_out(self.layers['decoder'](x))
            
        return x
    
    
    # get all weights in net
    def netWeightHist(self):
        # initialize weight vector
        W = np.array([])
        
        # get set of weights from each layer
        for layer in self.layers:
            W = np.concatenate((W, self.layers[f'{layer}'].weight.detach().flatten().numpy() ))
        
        # compute histogram
        bins = np.linspace(-.8, .8, 101)
        histy, histx = np.histogram(W, bins=bins, density=True)
        histx = (histx[1:] + histx[:-1])/2 # correct the dimension
        
        return histx, histy
    
    # get weight change (Frobenius)
    def netWeightFro(self, preWeight):
        
        # count params
        nParam = len([param_name for param_name, weight in self.named_parameters() if self.wFro in param_name])
        
        # init vars
        wChange = np.zeros(nParam)
        wConds  = np.zeros(nParam)
        
        i = 0
        for param_name, weight in self.named_parameters():
            
            if self.wFro in param_name:
                
                # Frobenius norm of the weight change from pre-training
                wConds[i] = np.linalg.cond(weight.data)
                
                # condition number
                wChange[i] = np.linalg.norm(preWeight[i] - weight.data.numpy(), ord='fro')
                
                # increment
                i += 1
            
            
        return wConds, wChange
    
    def trainModel(self,
                   DataLoader_train,
                   DataLoader_test,
                   epochs=1000,
                   loss_function=None,
                   comp_acc_test=None,
                   comp_w_hist=None,
                   comp_w_change=None
                   ):

        # ---------------------------------------
        # Metaparams
        # ---------------------------------------
        self.loss_func = loss_function
        self.compAccTest = comp_acc_test

        # loss function
        if loss_function == 'MSE':
            loss_fun = nn.MSELoss()  # mean squared error
            

        # ---------------------------------------
        # store results
        # ---------------------------------------

        # train set

        # initialize variables loss acc train set 
        losses_ep_train = torch.zeros(epochs)
        acc_ep_train = torch.zeros(epochs)

        # initialize variables acc test set
        acc_ep_test = torch.zeros(epochs)

        # initialize histogram variables
        histx = np.zeros((epochs, 100)) # 100 bins(bin size)
        histy = np.zeros((epochs, 100))

        # count params
        nParam = len([param_name for param_name, weight in self.named_parameters() if self.wFro in param_name])
        # initialize weight change var
        wChange = np.zeros((nParam, epochs))
        wConds  = np.zeros((nParam, epochs))


        # begin training the model
        for epoch in range(epochs):

            
            # compute weight change
            if comp_w_change == True:
                # store the weights for each layer
                preW = []
                for param_name, weight in self.named_parameters():
                    if self.wFro in param_name:
                        preW.append( copy.copy(weight.data.numpy()) )


            # compute weight histogram
            if comp_w_hist == True:
                histx, histy[epoch, :] = self.netWeightHist()


            #  switch training on, and implement dropout
            self.train()

            # batch
            losses_batch_train = torch.zeros(len(DataLoader_train))
            acc_batch_train = torch.zeros(len(DataLoader_train))

            batch = 0
            for X_batch_train, y_batch_true_train in DataLoader_train:

                # forward pass
                # predicted result from model
                y_pred_train = self.forward(X_batch_train)

                # compute loss
                # y = target/true value
                loss = loss_fun(y_pred_train, y_batch_true_train)

                # store loss in every batch
                losses_batch_train[batch] = loss

                # backpropagation
                self.optimizer.zero_grad()  # set derivative gradient of the model to be zero
                loss.backward()             # back propagation on the computed losses
                self.optimizer.step()       # stochastic gradient
                
                # batch increment
                batch += 1
                # end of batch training ----------------------------------------


            # average batch losses&acc
            losses_ep_train[epoch] = torch.mean(losses_batch_train)
            acc_ep_train[epoch] = torch.mean(acc_batch_train)

            if comp_acc_test == True:
                # compute accuracy per epoch of test set
                X_ep_test, y_ep_true_test = next(iter(DataLoader_test))
                _, acc_epoch_test = self.predict(X_ep_test, y_ep_true_test)
                acc_ep_test[epoch] = acc_epoch_test
                
            # weight change
            if comp_w_change == True:
                wConds[:, epoch] = self.netWeightFro(preW)[0] 
                wChange[:, epoch] = self.netWeightFro(preW)[1]
                
                
            # end of epoch training ----------------------------------------

        # total number of trainable parameters in the model
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        # save values weight change during training
        if comp_w_change == True:
            self.trainWChange = wChange
            self.trainWConds = wConds

        # end of function ----------------------------------------

        return y_pred_train, losses_ep_train, acc_ep_train, acc_ep_test, n_params, histx, histy

    def predict(self, data, y_true=None):

        # Make prediction
        self.eval()  # switch training off and no dropout during this mode

        with torch.no_grad():  # deactivates autograd
            predictions = self.forward(data)

        total_acc = 0
        if self.compAccTest == True:
            # Model Accuracy
            if self.loss_func == 'MSE':
                loss_fun = nn.MSELoss()
                total_acc = loss_fun(predictions, y_true)

            
        return predictions, total_acc