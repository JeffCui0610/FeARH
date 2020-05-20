# -*- coding: utf-8 -*-
"""
Created on Wed May 20 13:00:01 2020

@author: 86159
"""


import numpy as np
from keras.models import Sequential
from keras import regularizers
import keras.backend as K
import copy
import random
from keras.layers import Dense
from sklearn.metrics import average_precision_score,roc_auc_score



        

def getWeightsBias(mList,weightsList,biasList,numModel):
    for counterLayer in range(3):
        for counterModel in range(numModel):
            weightsList[counterLayer][counterModel]=mList[counterModel].layers[counterLayer].get_weights()[0]
            biasList[counterLayer][counterModel]=mList[counterModel].layers[counterLayer].get_weights()[1]
    return weightsList,biasList

def buildModelList(numUnitLayer0,numUnitLayer1,numUnitLayer2,xTrain,numModel):

    nn=Sequential([
            Dense(numUnitLayer0, activation='relu', input_dim=xTrain.shape[1], kernel_regularizer=regularizers.l2(0.01)),
            Dense(numUnitLayer1, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
            Dense(numUnitLayer2, activation='sigmoid')
        ])
    return nn

      
def aucrocScore(prediction,numModel,yTest):
    aucrocScore=np.ones((numModel))
    for aucrocCounter in range(numModel):
        aucrocScore[aucrocCounter]=roc_auc_score(yTest, prediction[aucrocCounter])
    return aucrocScore



'''this function using layers initial weights and bias to set weights and bias'''     
def iniWeightsBiasList2(numUnitLayer,numLayer,numModel,X):
    K.clear_session()
    nnInitial=Sequential([
        Dense(numUnitLayer[0], activation='relu', input_dim=X.shape[1], kernel_regularizer=regularizers.l2(0.01)),
        Dense(numUnitLayer[1], activation='relu', kernel_regularizer=regularizers.l2(0.01)),
        Dense(numUnitLayer[2], activation='sigmoid')
    ])
    weightsLayer0=np.ones((numModel,X.shape[1],numUnitLayer[0]))
    weightsLayer1=np.ones((numModel,numUnitLayer[0],numUnitLayer[1]))
    weightsLayer2=np.ones((numModel,numUnitLayer[1],numUnitLayer[2]))
    biasLayer0=np.ones((numModel,numUnitLayer[0]))
    biasLayer1=np.ones((numModel,numUnitLayer[1]))
    biasLayer2=np.ones((numModel,numUnitLayer[2]))
    weightsList,biasList=[weightsLayer0,weightsLayer1,weightsLayer2],[biasLayer0,biasLayer1,biasLayer2]
    for counterLayer in range(numLayer):
        for counterModel in range(numModel):
            weightsList[counterLayer][counterModel]=nnInitial.layers[counterLayer].get_weights()[0]
            biasList[counterLayer][counterModel]=nnInitial.layers[counterLayer].get_weights()[1]
    return weightsList, biasList
            
    

    
def shuffleData(x,y):
    '''remove patient names from the sheet, for later processing'''
    X=x.values[0:x.shape[0],1:x.shape[1]]
    Xcopy=copy.deepcopy(X)
    Y=y.values[0:x.shape[0],1]
    Ycopy=copy.deepcopy(Y)
    
    '''shuffle the data by changing te'''
    dataIndex=random.sample(range(X.shape[0]),X.shape[0])
    for counterDataShuffle in range(X.shape[0]):
        X[counterDataShuffle]=Xcopy[dataIndex[counterDataShuffle]]
        Y[counterDataShuffle]=Ycopy[dataIndex[counterDataShuffle]]
    print('data shuffle done')
    return X,Y



def averageParameter(sumDataPiece,numUnitLayer,X,numElement,numLayer,weightsGroupList,biasGroupList):
    '''Average parameters based on the percentage of data pieces used for training'''
    '''and the final result is parameters of just one neural net'''
    percent=sumDataPiece/sumDataPiece.sum()
    finalWeightsLayer0=np.zeros((X.shape[1],numUnitLayer[0]))
    finalWeightsLayer1=np.zeros((numUnitLayer[0],numUnitLayer[1]))
    finalWeightsLayer2=np.zeros((numUnitLayer[1],numUnitLayer[2]))
    finalWeightsList=[finalWeightsLayer0,finalWeightsLayer1,finalWeightsLayer2]
    finalBiasLayer0=np.zeros((numUnitLayer[0]))
    finalBiasLayer1=np.zeros((numUnitLayer[1]))
    finalBiasLayer2=np.zeros((numUnitLayer[2]))
    finalBiasList=[finalBiasLayer0,finalBiasLayer1,finalBiasLayer2]
    
    for counterList in range(numElement):
        for counterLayer in range(numLayer):
            finalWeightsList[counterLayer]=finalWeightsList[counterLayer]+percent[counterList]*weightsGroupList[counterList][counterLayer][0]
            finalBiasList[counterLayer]=finalBiasList[counterLayer]+percent[counterList]*biasGroupList[counterList][counterLayer][0]
    return finalWeightsList, finalBiasList
       
        
def initParameter2(numUnitLayer0,numUnitLayer1,numUnitLayer2,X,numModel,numUnitLayer,numLayer):
    nnInitParameter=buildModelList(numUnitLayer0,numUnitLayer1,numUnitLayer2,X,1)
    weightsLayer0=np.ones((numModel,X.shape[1],numUnitLayer[0]))
    weightsLayer1=np.ones((numModel,numUnitLayer[0],numUnitLayer[1]))
    weightsLayer2=np.ones((numModel,numUnitLayer[1],numUnitLayer[2]))
    biasLayer0=np.ones((numModel,numUnitLayer[0]))
    biasLayer1=np.ones((numModel,numUnitLayer[1]))
    biasLayer2=np.ones((numModel,numUnitLayer[2]))
    weightsGlobalList=[weightsLayer0,weightsLayer1,weightsLayer2]
    biasGlobalList=[biasLayer0,biasLayer1,biasLayer2]
    for counterLayer in range(numLayer):
        for counterModel in range(numModel):
            weightsGlobalList[counterLayer][counterModel]=nnInitParameter.layers[counterLayer].get_weights()[0]
            biasGlobalList[counterLayer][counterModel]=nnInitParameter.layers[counterLayer].get_weights()[1]
    return weightsGlobalList,biasGlobalList

def loadNNModel(numModel,nn,weightsGlobalList, biasGlobalList, counterNode, numLayer):
    for counterLayer in range(numLayer):
        wbLayer=[weightsGlobalList[counterLayer][counterNode],biasGlobalList[counterLayer][counterNode]]
        nn.layers[counterLayer].set_weights(wbLayer)
        
def randomOneZero(array, percent):
    num=array.shape[0]#for 1D array
    oneZeroArray=np.zeros((num))
    index=random.sample(range(num),int(num*percent))
    for counter in range(int(num*percent)):
        oneZeroArray[index[counter]]=1
    return oneZeroArray

def hybrid(numModel,weightsGlobalList, exchangeRate):
    index=random.sample(range(numModel), numModel)
    for counterHalfNode in range(int(numModel/2)):
        parameter0=weightsGlobalList[0][index[counterHalfNode]]
        #parameter0 is in size 2913*4
        parameter1=weightsGlobalList[0][index[counterHalfNode+1]]
        
        bitSelectIndex=randomOneZero(parameter0,exchangeRate)
        _1to0=np.transpose(parameter1*bitSelectIndex)
        _0to1=np.transpose(parameter0*bitSelectIndex)
        
        bitSelectIndex=-(bitSelectIndex-1)
        
        parameter0=np.transpose(parameter0*bitSelectIndex)
        parameter1=np.transpose(parameter1*bitSelectIndex)
        
        parameter0=parameter0+_1to0
        parameter1=parameter1+_0to1
    return weightsGlobalList

def averageParameter2(sumDataPiece,X,numUnitLayer,weightsGlobalList, biasGlobalList,numModel,numLayer):
    percent=sumDataPiece/sumDataPiece.sum()
    finalWeightsLayer0=np.zeros((X.shape[1],numUnitLayer[0]))
    finalWeightsLayer1=np.zeros((numUnitLayer[0],numUnitLayer[1]))
    finalWeightsLayer2=np.zeros((numUnitLayer[1],numUnitLayer[2]))
    finalWeightsList=[finalWeightsLayer0,finalWeightsLayer1,finalWeightsLayer2]
    finalBiasLayer0=np.zeros((numUnitLayer[0]))
    finalBiasLayer1=np.zeros((numUnitLayer[1]))
    finalBiasLayer2=np.zeros((numUnitLayer[2]))
    finalBiasList=[finalBiasLayer0,finalBiasLayer1,finalBiasLayer2]
    
    for counterModel in range(numModel):
        for counterLayer in range(numLayer):
            finalWeightsList[counterLayer]=finalWeightsList[counterLayer]+percent[counterModel]*weightsGlobalList[counterLayer][counterModel]
            finalBiasList[counterLayer]=finalBiasList[counterLayer]+percent[counterModel]*biasGlobalList[counterLayer][counterModel]
    
    return finalWeightsList,finalBiasList

def flatModel(numParameters, numLayerValuesList,weightsGlobalList, biasGlobalList,modelIndex):
    flatParameter=np.zeros((numParameters))
    flatParameter[0:numLayerValuesList[0]]=weightsGlobalList[0][modelIndex].flatten()
    flatParameter[numLayerValuesList[0]:numLayerValuesList[1]]=weightsGlobalList[1][modelIndex].flatten()
    flatParameter[numLayerValuesList[1]:numLayerValuesList[2]]=weightsGlobalList[2][modelIndex].flatten()
    flatParameter[numLayerValuesList[2]:numLayerValuesList[3]]=biasGlobalList[0][modelIndex].flatten()
    flatParameter[numLayerValuesList[3]:numLayerValuesList[4]]=biasGlobalList[1][modelIndex].flatten()
    flatParameter[numLayerValuesList[4]:numLayerValuesList[5]]=biasGlobalList[2][modelIndex].flatten()
    return flatParameter
    
        
    
def reconstructModel(flatParameter,numLayerValuesList,weightsGlobalList,biasGlobalList,modelIndex):
    weightsGlobalList[0][modelIndex]=flatParameter[0:numLayerValuesList[0]].reshape((2913,4))
    weightsGlobalList[1][modelIndex]=flatParameter[numLayerValuesList[0]:numLayerValuesList[1]].reshape((4,2))
    weightsGlobalList[2][modelIndex]=flatParameter[numLayerValuesList[1]:numLayerValuesList[2]].reshape((2,1))
    biasGlobalList[0][modelIndex]=np.transpose(flatParameter[numLayerValuesList[2]:numLayerValuesList[3]])
    biasGlobalList[1][modelIndex]=np.transpose(flatParameter[numLayerValuesList[3]:numLayerValuesList[4]])
    biasGlobalList[2][modelIndex]=np.transpose(flatParameter[numLayerValuesList[4]:numLayerValuesList[5]])
    
    
    
    

def hybrid2(numUnitLayer,xLocalTrain,numModel,exchangeRate,weightsGlobalList, biasGlobalList):
    numWeightsLayer0=xLocalTrain.shape[1]*numUnitLayer[0]
    numWeightsLayer1=numUnitLayer[0]*numUnitLayer[1]+numWeightsLayer0
    numWeightsLayer2=numUnitLayer[1]*numUnitLayer[2]+numWeightsLayer1
    numBiasLayer0=numUnitLayer[0]+numWeightsLayer2
    numBiasLayer1=numUnitLayer[1]+numBiasLayer0
    numBiasLayer2=numUnitLayer[2]+numBiasLayer1
    numLayerValuesList=[numWeightsLayer0,numWeightsLayer1,numWeightsLayer2,
                       numBiasLayer0,numBiasLayer1,numBiasLayer2]
    numParameters=numWeightsLayer0+numWeightsLayer1+numWeightsLayer2+numBiasLayer0+numBiasLayer1+numBiasLayer2#this variable represent the number of all parameters in a neural network
    indexModel=random.sample(range(numModel), numModel)
    for counterHalfModel in range(int(numModel/2)):
        flatParameter0=flatModel(numParameters, numLayerValuesList,
                                 weightsGlobalList, biasGlobalList,indexModel[counterHalfModel*2])
        flatParameter1=flatModel(numParameters, numLayerValuesList,
                                 weightsGlobalList, biasGlobalList,indexModel[counterHalfModel*2+1])
#        indexParameter=random.sample(range(numParameters),numParameters*exchangeRate)
        '''Build an array with the same size of flatParameter'''
        '''This is used for values selection'''

        bitSelectIndex=randomOneZero(flatParameter1,exchangeRate)

        _1to0=flatParameter1*bitSelectIndex
        _0to1=flatParameter0*bitSelectIndex
        
        bitSelectIndex=-(bitSelectIndex-1)

        
        flatParameter0=flatParameter0*bitSelectIndex
        flatParameter1=flatParameter1*bitSelectIndex
        
        
        flatParameter0=flatParameter0+_1to0
        flatParameter1=flatParameter1+_0to1
        

        reconstructModel(flatParameter0,numLayerValuesList,
                         weightsGlobalList,biasGlobalList,indexModel[counterHalfModel*2])
        
        reconstructModel(flatParameter1,numLayerValuesList,
                         weightsGlobalList,biasGlobalList,indexModel[counterHalfModel*2+1])
        
    return weightsGlobalList,biasGlobalList    
        
    
            


