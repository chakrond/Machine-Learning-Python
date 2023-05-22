# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 16:34:41 2022

@author: Chakron.D
"""
# %% Importing the libraries

# numpy
import numpy as np
import copy
import math

# PyTroch
import torch
import torch.nn as nn
import torch.nn.functional as F

# loss functions
import Function.LossFunc as LF

# %% CNN Class


class CNN_Class(nn.Module):

    # Inheritance from parents class
    def __init__(self):
        super().__init__()
        # super(ANN_Class, self).__init__()
    
    
    # set layers structures
    def setLayers(self,
                  imgSize,
                  convLayer,
                  poolLayer,
                  convDecLayer,
                  outputLayer,
                  batch_norm=None,
                  printToggle=False
                  ):
        
        # ---------------------------------------
        # Parameters
        # ---------------------------------------
        self.poolLayer = poolLayer
        self.batchNorm = batch_norm
        self.printToggle = printToggle
        self.nConvLayer = len(convLayer)
        self.nConvDecLayer = len(convDecLayer)
        
        
        # ---------------------------------------
        # Layer structure
        # ---------------------------------------
        self.layers = nn.ModuleDict()  # store layers

        # Convolution layers
        convLayer_names = list(convLayer.keys())
        
        # Convolution Decoding layers
        convDecLayer_names = list(convDecLayer.keys())
        
        # output layers
        outputLayer_names = list(outputLayer.keys())        


        # Encoding layers
        for i in range(len(convLayer)):
            
            # Create layers
            self.layers[f'conv{i}'] = nn.Conv2d(**convLayer[convLayer_names[i]])
            
            
        # Decoding layers
        for i in range(len(convDecLayer)):
            
            # Create layers
            self.layers[f'convDec{i}'] = nn.ConvTranspose2d(**convDecLayer[convDecLayer_names[i]])
            
        # output layer
        self.layers['output'] = nn.ConvTranspose2d(**outputLayer[outputLayer_names[0]])
        
    
    def outputSize(self, inputSize, kernelSize, stride, padd):
        
        size = np.floor( (inputSize+2*padd-kernelSize)/stride )+1
        
        return size
        
    def setParams(self,
                 dropout_rate=0.5,
                 learning_rate=0.01,
                 w_decay=None,
                 p_lambda=0.01,
                 p_momentum=0,
                 act_lib='torch',
                 conv_activation_fun='relu',  # activation function at convolution layers (fully-connected layers)
                 pool_lib='torch.nn.functional',
                 pool_fun='max_pool2d',
                 activation_fun_out='sigmoid',
                 optim_fun='Adam',
                 save_FeatMap=False,
                 lr_decay=None,
                 lr_step_size=None,
                 lr_gamma=None,
                 w_fro=''
                 ):

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
        # activaiton funciton
        self.actLib = act_lib
        self.convActFun = conv_activation_fun
        self.actfunOut = activation_fun_out
        # pooling
        self.poolLib = pool_lib
        self.poolFun = pool_fun
        # Learning rate decay bool
        self.lrDecay = lr_decay
        # weight params
        self.wFro = w_fro
        # save feature maps
        self.saveFeatMap = save_FeatMap

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

        # Activation Function
        convActFun = getattr(eval(self.actLib), self.convActFun)

        # Pooling Function
        poolingFun = getattr(eval(self.poolLib), self.poolFun)
        
        # Activation functions decoder
        actFun_out = getattr(torch, self.actfunOut)
        
        
        # **--Start Forward Prop--**
        
        if self.printToggle: print(f'Input: {x.shape}')
        
        # **--Encoding Convolution layers--**
        poolLayers_name = list(self.poolLayer.keys())
        
        # store results from convolution
        if self.saveFeatMap == True:
            self.featureMaps = {}
        
        for i in range(self.nConvLayer):
            
            # convolution -> pooling -> activation
            if self.saveFeatMap == True:
                x = convActFun(self.layers[f'conv{i}'](x))
                self.featureMaps[f'conv{i}'] = x.clone().detach()
                x = poolingFun(x, **self.poolLayer[poolLayers_name[i]])
            else:
                x = convActFun( poolingFun(self.layers[f'conv{i}'](x), **self.poolLayer[poolLayers_name[i]]) )
            
            if self.printToggle: print(f'Layer conv{i}/pool{i}: {x.shape}')


        # **--Decoding Convolution layers--**
        
        for i in range(self.nConvDecLayer):
            
            # convolution -> pooling -> activation
            x = convActFun(self.layers[f'convDec{i}'](x))
            if self.saveFeatMap == True: self.featureMaps[f'convDec{i}'] = x.clone().detach()

            if self.printToggle: print(f'Layer convDec{i}/pool{i}: {x.shape}')


        # **--Output layers--**
        x = actFun_out(self.layers['output'](x))
        
        if self.printToggle: print(f'Layer output: {x.shape}')


        return x
    
    
    # get all weights in net
    def netWeightHist(self):
        # initialize weight vector
        W = np.array([])
        
        # get set of weights from each layer
        for layer in self.layers:
            layer_W = self.layers[f'{layer}'].weight.cpu() if self.device else self.layers[f'{layer}'].weight
            W = np.concatenate((W, layer_W.detach().flatten().numpy() ))
        
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
                   comp_w_change=None,
                   device=None
                   ):
        
        # Device
        self.device = device

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
            loss_fun = nn.BCEWithLogitsLoss()  # already combines a **Sigmoid layer

        if loss_function == 'MSE':
            loss_fun = nn.MSELoss()  # mean squared error
            
        if loss_function == 'NLL':
            loss_fun = nn.NLLLoss() # negative log likelihood loss
            
        if loss_function == 'mLoss':
            loss_fun = LF.mLoss()

        if loss_function == 'ConjLoss':
            loss_fun = LF.ConjLoss()

        if loss_function == 'CorrLoss':
            loss_fun = LF.CorrLoss()

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

                if device:
                    # data to GPU
                    X_batch_train = X_batch_train.to(device)
                    y_batch_true_train = y_batch_true_train.to(device)


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
                # end of batch training ----------------------------------------

            # average batch losses&acc
            losses_ep_train[epoch] = torch.mean(losses_batch_train)
            acc_ep_train[epoch] = torch.mean(acc_batch_train)

            if comp_acc_test == True:
                # compute accuracy per epoch of test set
                X_ep_test, y_ep_true_test = next(iter(DataLoader_test))
                
                if device:
                    # data to GPU
                    X_ep_test = X_ep_test.to(device)
                    y_ep_true_test = y_ep_true_test.to(device)
                
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