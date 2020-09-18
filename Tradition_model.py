
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import keras.backend as K
from keras import regularizers
import FeARH_lib as ga
from sklearn.metrics import roc_auc_score,average_precision_score,f1_score



def train_target(X,Y):
    '''the number of node in each layer in neural network '''
    numUnitLayer0=4
    numUnitLayer1=2
    numUnitLayer2=1
    numLayer=3
    numUnitLayer=[numUnitLayer0,numUnitLayer1,numUnitLayer2]#improve convenience for later iteration

    numModel=8##
    numNode=8
    first_parameter_store=[]
    '''initial testset'''
    xFinalTest=X[25000:]#####################################################

    '''Initial parameter'''
    '''every model have the same parameter at the beginning'''
    weightsGlobalList,biasGlobalList=ga.initParameter2(numUnitLayer0,numUnitLayer1,numUnitLayer2,
                                                       X,numModel,numUnitLayer,numLayer)
    
    train_node_size=scala
    sumDataPiece=scala*np.ones((numNode))
    
    
    current_aucroc_score=0.01
    previous_aucroc_score=0.01
    
    halting_condition=0.0001
    
    
    
    aucroc_store=[]
    parameter_store=[] 
    stop_counter=0
    num_hybrid=0
    while stop_counter<3:
        num_hybrid+=1       
        for i in range(numNode):
            xTrain=X[scala*i:train_node_size*(i+1)]
            yTrain=Y[train_node_size*i:train_node_size*(i+1)]
            K.clear_session()
            nn=ga.buildModelList(numUnitLayer0,numUnitLayer1,numUnitLayer2,X,1)
            ga.loadNNModel(numModel,nn,weightsGlobalList, biasGlobalList, i, numLayer)
            nn.compile(optimizer='Nadam', loss='binary_crossentropy')
            nn.fit(xTrain, yTrain,nb_epoch=10, verbose=0) 
            for counterLayer in range(numLayer):
                weightsGlobalList[counterLayer][i]=nn.layers[counterLayer].get_weights()[0]
                biasGlobalList[counterLayer][i]=nn.layers[counterLayer].get_weights()[1]
        finalWeightsList, finalBiasList=ga.averageParameter2(sumDataPiece,X,numUnitLayer,
                                                            weightsGlobalList, biasGlobalList,numModel,numLayer)
        '''Update weightsGlobalList and biasGlobalList with averaged parameters'''
    #    weightsGlobalList, biasGlobalList=[],[]
        weightsGlobalList=[np.array([finalWeightsList[idx]]*numNode) for idx in [0,1,2]]
        biasGlobalList=[np.array([finalBiasList[idx]]*numNode) for idx in [0,1,2]]
        
        '''Retrieve the first model's parameter'''
#        first_model = ga.buildModelList(numUnitLayer0,numUnitLayer1,numUnitLayer2,xFinalTest,numModel)
        temp_first_parameter_store = []
        for counterLayer in range(numLayer):
            wbLayer1=[weightsGlobalList[counterLayer][0],biasGlobalList[counterLayer][0]]
            temp_first_parameter_store.append(wbLayer1)
        first_parameter_store.append(temp_first_parameter_store)
        
        
        
        finalModel=ga.buildModelList(numUnitLayer0,numUnitLayer1,numUnitLayer2,xFinalTest,numModel)
        temp_paramter_store=[]
        for counterLayer in range(numLayer):
            wbLayer=[finalWeightsList[counterLayer],finalBiasList[counterLayer]]
            temp_paramter_store.append(wbLayer)
            finalModel.layers[counterLayer].set_weights(wbLayer)
        parameter_store.append(temp_paramter_store)
        
        
        '''This part is for validation in distributed manner'''
        validation_silo_size=250
        validation_aucroc=[]
        validation_acupr=[]
        base_index=numNode*scala
        for i in range(numNode):
            validation_x=X[base_index+(i*validation_silo_size):base_index+((i+1)*validation_silo_size)]#
            validation_y=Y[base_index+(i*validation_silo_size):base_index+((i+1)*validation_silo_size)]
            pred_validation=finalModel.predict(validation_x)
            aucrocScore_validation=roc_auc_score(validation_y, pred_validation)
            aucprScore_validation=average_precision_score(validation_y, pred_validation)
            validation_aucroc.append(aucrocScore_validation)
            validation_acupr.append(aucprScore_validation)
        current_aucroc_score=np.mean(validation_aucroc)
        
        if (current_aucroc_score-previous_aucroc_score)/previous_aucroc_score<=halting_condition:
            stop_counter+=1
        else:
            stop_counter=0
        aucroc_store.append(current_aucroc_score)
        previous_aucroc_score=max(aucroc_store)
        
    
#    '''Build model for testing'''
#    K.clear_session()
#    parameter_index=aucroc_store.index(max(aucroc_store))
#    test_parameter=parameter_store[parameter_index]
#    test_model=ga.buildModelList(numUnitLayer0,numUnitLayer1,numUnitLayer2,xFinalTest,numModel)
#    for counterLayer in range(numLayer):
#        test_model.layers[counterLayer].set_weights(test_parameter[counterLayer])
#    return test_model
    """Build the first model"""
    parameter_index=aucroc_store.index(max(aucroc_store))
    test_parameter1=first_parameter_store[parameter_index]
    test_model1=ga.buildModelList(numUnitLayer0,numUnitLayer1,numUnitLayer2,xFinalTest,numModel)
    for counterLayer in range(numLayer):
        test_model1.layers[counterLayer].set_weights(test_parameter1[counterLayer])
    return test_model1


def target_model_fn():
    """The architecture of the target (victim) model.
    The attack is white-box, hence the attacker is assumed to know this architecture too."""

    model = tf.keras.models.Sequential()

    model.add(layers.Dense(4, activation="relu", input_dim=2913, kernel_regularizer=regularizers.l2(0.01)))
    model.add(layers.Dense(2, activation="relu", kernel_regularizer=regularizers.l2(0.01)))
    model.add(layers.Dense(1, activation="sigmoid"))
    model.compile(optimizer='Nadam', loss='binary_crossentropy')
    return model

scala = 2000 #Number of records in each node

x=pd.read_csv(r"...\drug_table_mortality_with_no.csv")
y=pd.read_csv(r"...\mortality_table.csv")

X,Y=ga.shuffleData(x,y)

target_model = train_target(X,Y) # Reture the model of traditional model



