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
                  hiddLayer,
                  feature_out,
                  conv_batch_norm=None,
                  batch_norm=None,
                  printToggle=False
                  ):
        
        # ---------------------------------------
        # Parameters
        # ---------------------------------------
        # Conv layer
        self.poolLayer = poolLayer
        self.convBatchNorm = conv_batch_norm
        self.nConvLayer = len(convLayer)
        
        # FF layer
        self.nHLayer = len(hiddLayer)
        self.batchNorm = batch_norm
        
        # Print
        self.printToggle = printToggle

        # ---------------------------------------
        # Layer structure
        # ---------------------------------------
        self.layers = nn.ModuleDict()  # store layers

        # Convolution layers
        convLayer_names = list(convLayer.keys())
        
        # Maxpool layers
        poolLayer_names = list(poolLayer.keys())
        
        # Hidden layers
        hiddLayer_names = list(hiddLayer.keys())
        
        # Output size
        output_size = []

        for i in range(len(convLayer)):
            
            # Create layers
            self.layers[f'conv{i}'] = nn.Conv2d(**convLayer[convLayer_names[i]])
            
            # output size calculation
            if i == 0:
                size = self.outputSize(imgSize, 
                                        convLayer[convLayer_names[i]]['kernel_size'], 
                                        convLayer[convLayer_names[i]]['stride'],
                                        convLayer[convLayer_names[i]]['padding']
                                        )/poolLayer[poolLayer_names[i]]['kernel_size']
                output_size.append( np.floor(size) )
                
                # Batch normalize
                if conv_batch_norm == True:
                    self.layers[f'convBatchNorm{i}'] = nn.BatchNorm2d(convLayer[convLayer_names[i]]['out_channels'])
                
            else:
                size = self.outputSize(output_size[i-1], 
                                        convLayer[convLayer_names[i]]['kernel_size'], 
                                        convLayer[convLayer_names[i]]['stride'],
                                        convLayer[convLayer_names[i]]['padding']
                                        )/poolLayer[poolLayer_names[i]]['kernel_size']
                output_size.append( np.floor(size) )
                
                # Batch normalize
                if conv_batch_norm == True:
                    self.layers[f'convBatchNorm{i}'] = nn.BatchNorm2d(convLayer[convLayer_names[i]]['out_channels'])
                    

        # Expected size before get into Fully-connected layers
        # Fully-connected layers has no kernel, padding or stride
        expSize = self.outputSize(output_size[-1], 1 , 1, 0)
        
        expSize = convLayer[convLayer_names[-1]]['out_channels']*int(expSize**2)


        # Fully-connected layers
        
        self.layers['input'] = nn.Linear(expSize, hiddLayer[hiddLayer_names[0]]['in_features'])  # input layer

        for i in range(self.nHLayer):
            self.layers[f'hidden{i}'] = nn.Linear(**hiddLayer[hiddLayer_names[i]])  # hidden layers

            if batch_norm == True:
                self.layers[f'batchNorm{i}'] = nn.BatchNorm1d(hiddLayer[hiddLayer_names[i]]['out_features'])

        self.layers['output'] = nn.Linear(hiddLayer[hiddLayer_names[-1]]['out_features'], feature_out)  # output layer
        
    
    def outputSize(self, inputSize, kernelSize, stride, padd):
        
        size = np.floor( (inputSize+2*padd-kernelSize)/stride )+1
        
        return size
        
    def setParams(self,
                 conv_dropout_rate=0,
                 dropout_rate=0,
                 learning_rate=0.01,
                 w_decay=None,
                 p_lambda=0.01,
                 p_momentum=0,
                 act_lib='torch',
                 conv_activation_fun='relu',  # activation function at convolution layers
                 pool_lib='torch.nn.functional',
                 pool_fun='max_pool2d',
                 activation_fun='relu',  # activation function at hidden layers (fully-connected layers)
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
        self.convdr = conv_dropout_rate # Conv layer
        self.dr = dropout_rate # FF layer
        # Learning rate
        self.lr = learning_rate
        # weight decay
        self.wDecay = w_decay
        # Lambda
        self.Ld = p_lambda
        # activaiton funciton
        self.actLib = act_lib
        self.actfun = activation_fun
        self.convActFun = conv_activation_fun
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
        actFun = getattr(eval(self.actLib), self.actfun)
        convActFun = getattr(eval(self.actLib), self.convActFun)

        # Pooling Function
        poolingFun = getattr(eval(self.poolLib), self.poolFun)
        
        # **--Start Forward Prop--**
        
        if self.printToggle: print(f'Input: {x.shape}')
        
        # **--Convolution layers--**
        poolLayers_name = list(self.poolLayer.keys())
        
        # store results from convolution
        if self.saveFeatMap == True:
            self.featureMaps = {}
        
        for i in range(self.nConvLayer):
            
        # convolution -> pooling ->  batchnorm -> activation
        
            # convolution
            x = self.layers[f'conv{i}'](x)
            
            if self.saveFeatMap == True:
                self.featureMaps[f'conv{i}'] = x.clone().detach()
            
            # pooling
            x = poolingFun(x, **self.poolLayer[poolLayers_name[i]])
            
            # bacth noarmalize
            if self.convBatchNorm == True:
                x = self.layers[f'convBatchNorm{i}'](x)
            
            # activation
            x = convActFun(x)
            
            # dropout 
            x = F.dropout(x, p=self.convdr, training=self.training)
                    
            if self.printToggle: print(f'Layer conv{i}/pool{i}: {x.shape}')

        # Reshape for linear layer - Vectorize
        nUnits = x.shape.numel()/x.shape[0]
        x = x.view(-1, int(nUnits))
        if self.printToggle: print(f'Vectorize: {x.shape}')


        # **--Fully-connected layers--**
        
        # input layer
        x = self.layers['input'](x)
        if self.printToggle: print(f'FC_input: {x.shape}')

        # dropout after input layer
        # training=self.training means to turn off during eval mode
        x = F.dropout(x, p=self.dr, training=self.training)


        # hidden layers
        for i in range(self.nHLayer):

            if self.batchNorm == True:
                x = self.layers[f'batchNorm{i}'](x)

            if self.actLib == 'torch':
                # hidden layer
                # x = F.relu( self.layers[f'hidden{i}'](x) )
                x = actFun(self.layers[f'hidden{i}'](x))

            if self.actLib == 'torch.nn':
                # hidden layer
                x = actFun()(self.layers[f'hidden{i}'](x))
                
            if self.printToggle: print(f'FC_hidden{i}: {x.shape}')

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
        if loss_function == 'cross-entropy':
            # already include computing Softmax activation function at the output layer
            loss_fun = nn.CrossEntropyLoss()

        if loss_function == 'binary':
            loss_fun = nn.BCEWithLogitsLoss()  # already combines a **Sigmoid layer

        if loss_function == 'MSE':
            loss_fun = nn.MSELoss()  # mean squared error
            
        if loss_function == 'NLL':
            loss_fun = nn.NLLLoss() # negative log likelihood loss 

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