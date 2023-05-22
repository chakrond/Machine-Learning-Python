# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 16:34:41 2022

@author: Chakron.D
"""
# %% Importing the libraries

# numpy
import numpy as np

# PyTroch
import torch
import torch.nn as nn
import torch.nn.functional as F

# %% FNN Class


class FFN_Class(nn.Module):

    # Inheritance from parents class
    def __init__(self):
        super().__init__()
        # super(ANN_Class, self).__init__()
    
    
    def setParams(self,
                 feature_in,
                 feature_out,
                 n_hUnit,
                 n_hLayer,
                 dropout_rate=0.5,
                 learning_rate=0.01,
                 w_decay=None,
                 p_lambda=0.01,
                 p_momentum=0,
                 batch_norm=None,
                 act_lib='torch',
                 activation_fun='relu',  # activation function at hidden layers
                 optim_fun='Adam',
                 lr_decay=None,
                 lr_step_size=None,
                 lr_gamma=None,
                 ):


        # ---------------------------------------
        # Layer structure
        # ---------------------------------------
        self.layers = nn.ModuleDict()  # store layers

        self.n_hLayer = n_hLayer  # number of hidden layers

        self.layers['input'] = nn.Linear(feature_in, n_hUnit)  # input layer

        for i in range(n_hLayer):
            self.layers[f'hidden{i}'] = nn.Linear(
                n_hUnit, n_hUnit)  # hidden layers

            if batch_norm == True:
                self.layers[f'batchNorm{i}'] = nn.BatchNorm1d(n_hUnit)

        self.layers['output'] = nn.Linear(n_hUnit, feature_out)  # output layer

        # ---------------------------------------
        # Parameters
        # ---------------------------------------
        # Dropout rate
        self.dr = dropout_rate
        # Learning rate
        self.lr = learning_rate
        # weight decay
        self.wDecay = w_decay
        # Lambda
        self.Ld = p_lambda
        # batch Normalize bool
        self.batchNorm = batch_norm
        # activaiton funciton
        self.actLib = act_lib
        self.actfun = activation_fun
        # Learning rate decay bool
        self.lrDecay = lr_decay

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
          'weight_decay': p_lambda,
          'momentum': p_momentum
        }
        
        if optim_fun == 'Adam': 
            del paramsDict['momentum'] # delete momentum paramater as Adam optim doesn't have
        
        if w_decay == 'L2':
            self.optimizer = optimFun(self.parameters(), **paramsDict)  # stochastic gradient
        else:
            del paramsDict['weight_decay']
            self.optimizer = optimFun(self.parameters(), **paramsDict)

        # Learning rate decay
        if lr_decay == True:
            params_lrDecay_Dict = {
              'step_size': lr_step_size,
              'gamma': lr_gamma,
            }
            self.lrScheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, **params_lrDecay_Dict)


    # forward pass
    def forward(self, x):

        # input layer
        x = self.layers['input'](x)

        # dropout after input layer
        # training=self.training means to turn off during eval mode
        x = F.dropout(x, p=self.dr, training=self.training)

        if self.actLib == 'torch':
            # activation functions
            actFun = getattr(torch, self.actfun)

        if self.actLib == 'torch.nn':
            # activation functions
            actFun = getattr(torch.nn, self.actfun)

        # hidden layers
        for i in range(self.n_hLayer):

            if self.batchNorm == True:
                x = self.layers[f'batchNorm{i}'](x)

            if self.actLib == 'torch':
                # hidden layer
                # x = F.relu( self.layers[f'hidden{i}'](x) )
                x = actFun(self.layers[f'hidden{i}'](x))

            if self.actLib == 'torch.nn':
                # hidden layer
                x = actFun()(self.layers[f'hidden{i}'](x))

            # dropout
            x = F.dropout(x, p=self.dr, training=self.training)

        # return output layer
        if self.loss_func == 'NLL':
            x = torch.log_softmax(self.layers['output'](x) ,axis=1)
        else:
            x = self.layers['output'](x)
            
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
    
    
    def trainModel(self,
                   DataLoader_train,
                   DataLoader_test,
                   epochs=1000,
                   loss_function=None,
                   comp_acc_test=None,
                   comp_w_hist=None,
                   ):

        # ---------------------------------------
        # Metaparams
        # ---------------------------------------
        self.loss_func = loss_function
        self.compAccTest = comp_acc_test

        # loss function
        if loss_function == 'cross-entropy':
            # already include computing Softmax activation function at the output layer
            loss_fun = nn.CrossEntropyLoss()

        if loss_function == 'binary':
            loss_fun = nn.BCEWithLogitsLoss()  # already combines a Sigmoid layer

        if loss_function == 'MSE':
            loss_fun = nn.MSELoss()  # mean squared error
            
        if loss_function == 'NLL':
            loss_fun = nn.NLLLoss() # mean squared error    

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

        # begin training the model
        for epoch in range(epochs):

            # comput weight histogram
            if comp_w_hist == True:
                histx, histy[epoch,:] = self.netWeightHist()


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

                # if L1 is identified
                if self.wDecay == 'L1':
                    # add L1 term
                    L1_term = torch.tensor(0., requires_grad=True)
                    # sum up all abs(weights)
                    for param_name, weight in self.named_parameters():
                        if 'bias' not in param_name:
                            L1_term = L1_term + torch.sum(torch.abs(weight))
                    # add to loss term
                    loss = loss + self.Ld*L1_term/self.nWeights

                # store loss in every batch
                losses_batch_train[batch] = loss

                # backpropagation
                self.optimizer.zero_grad()  # set derivative gradient of the model to be zero
                loss.backward()             # back propagation on the computed losses
                self.optimizer.step()       # stochastic gradient

                # Learning rate decay
                if self.lrDecay == True:
                    self.lrScheduler.step()

                # compute accuracy per batch of training set
                if loss_function == 'cross-entropy':
                    # pick the highest probability and compare to true labels
                    y_pred_batch_train = torch.argmax(y_pred_train, axis=1) == y_batch_true_train
                    acc_ba_train = 100 * torch.sum(y_pred_batch_train.float()) / len(y_pred_batch_train)
                    acc_batch_train[batch] = acc_ba_train

                if loss_function == 'binary':
                    # pick the probability>0.5 and compare to true labels # 100*torch.mean(((predictions>0.5) == labels).float())
                    y_pred_batch_train = ((y_pred_train > 0.5) == y_batch_true_train).float()
                    acc_ba_train = 100 * torch.sum(y_pred_batch_train.float()) / len(y_pred_batch_train)
                    acc_batch_train[batch] = acc_ba_train
                    
                if loss_function == 'NLL':
                    # pick the highest probability and compare to true labels
                    y_pred_batch_train = torch.argmax(y_pred_train, axis=1) == y_batch_true_train
                    acc_ba_train = 100 * torch.sum(y_pred_batch_train.float()) / len(y_pred_batch_train)
                    acc_batch_train[batch] = acc_ba_train

                # batch increment
                batch += 1
                # ----------------------------------------

            # average batch losses&acc
            losses_ep_train[epoch] = torch.mean(losses_batch_train)
            acc_ep_train[epoch] = torch.mean(acc_batch_train)

            if comp_acc_test == True:
                # compute accuracy per epoch of test set
                X_ep_test, y_ep_true_test = next(iter(DataLoader_test))
                _, acc_epoch_test = self.predict(X_ep_test, y_ep_true_test)
                acc_ep_test[epoch] = acc_epoch_test
                
                
            # ----------------------------------------

        # total number of trainable parameters in the model
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        # ----------------------------------------

        return y_pred_train, losses_ep_train, acc_ep_train, acc_ep_test, n_params, histx, histy

    def predict(self, data, y_true=None):

        # Make prediction
        self.eval()  # switch training off and no dropout during this mode

        with torch.no_grad():  # deactivates autograd
            predictions = self.forward(data)

        total_acc = 0
        if self.compAccTest == True:
            # Model Accuracy
            if self.loss_func == 'cross-entropy':
                labels_pred = torch.argmax(predictions, axis=1) == y_true
                total_acc = 100*torch.sum(labels_pred.float()) / len(labels_pred)
    
            if self.loss_func == 'binary':
                labels_pred = ((predictions > 0.5) == y_true).float()
                total_acc = 100*torch.sum(labels_pred.float()) / len(labels_pred)
    
            if self.loss_func == 'MSE':
                loss_fun = nn.MSELoss()
                total_acc = loss_fun(predictions, y_true)
    
            if self.loss_func == 'NLL':
                labels_pred = torch.argmax(predictions, axis=1) == y_true
                total_acc = 100*torch.sum(labels_pred.float()) / len(labels_pred)
            
        return predictions, total_acc