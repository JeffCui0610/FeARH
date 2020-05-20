

import numpy as np
import pandas as pd
import keras.backend as K
import FeARH_lib as ga
from sklearn.metrics import average_precision_score,roc_auc_score



'''read file form csv file'''
x=pd.read_csv(r"C:\document\education\python\python\predict mortality\data\drug_table_mortality_with_no.csv")
y=pd.read_csv(r"C:\document\education\python\python\predict mortality\data\mortality_table.csv")



'''shuffle the dataset by changing the order between rows and rows randomly'''
X,Y=ga.shuffleData(x,y)

'''the number of node in each layer in neural network '''
numUnitLayer0=4
numUnitLayer1=2
numUnitLayer2=1
numLayer=3
numUnitLayer=[numUnitLayer0,numUnitLayer1,numUnitLayer2]

'''Get the total number of features'''
numFeature=X.shape[1]

'''initial testset (not necessary)'''
xFinalTest=X[25000:]
yFinalTest=Y[25000:]

numModel=8
numNode=8

step=260#the number of data pieces in each sub dataset


ratePerformance=[]
aucpr=[]

sumDataPiece=step*np.ones((numNode))
numHybridCycle=8
'''Initialize parameters'''
weightsGlobalList,biasGlobalList=ga.initParameter2(numUnitLayer0,numUnitLayer1,numUnitLayer2,
                                                           X,numModel,numUnitLayer,numLayer)



stop_counter=0
num_hybrid=0

'''Provide a small value as base line'''
current_aucroc_score=0.01
previous_aucroc_score=0.01

halting_condition=0.0001
aucroc_store=[]

result=[]

result_distributedtesting=[]

exchangeRate=0.1
aucroc_store=[]
parameter_store=[] 
stop_counter=0
while stop_counter<3:
        '''Here we start a hybridization cycle of iteration over all eight nodes'''
        for counterNode in range(numNode):
            '''Define local training data'''
            xLocalTrain=X[step*counterNode:step*(counterNode+1)]
            yLocalTrain=Y[step*counterNode:step*(counterNode+1)]
            K.clear_session()
            '''distribute local model'''
            nn=ga.buildModelList(numUnitLayer0,numUnitLayer1,numUnitLayer2,X,1)
            '''Load parameters in local node'''
            ga.loadNNModel(numModel,nn,weightsGlobalList, biasGlobalList, counterNode, numLayer)
            '''Using Gradient descent to train this model'''
            nn.compile(optimizer='Nadam', loss='binary_crossentropy')
            nn.fit(xLocalTrain, yLocalTrain,nb_epoch=10, verbose=0) 
            
            '''retrieve trained parameters back to matrics'''
            for counterLayer in range(numLayer):
                weightsGlobalList[counterLayer][counterNode]=nn.layers[counterLayer].get_weights()[0]
                biasGlobalList[counterLayer][counterNode]=nn.layers[counterLayer].get_weights()[1]
            '''Hybrid among all nodes in pairs'''
        weightsGlobalList,biasGlobalList =ga.hybrid2(numUnitLayer,xLocalTrain,numModel,
                             exchangeRate,weightsGlobalList, biasGlobalList)
        '''federate parameters by averaging'''
        finalWeightsList,finalBiasList=ga.averageParameter2(sumDataPiece,X,numUnitLayer,
                                                            weightsGlobalList, biasGlobalList,numModel,numLayer)
        finalModel=ga.buildModelList(numUnitLayer0,numUnitLayer1,numUnitLayer2,xFinalTest,numModel)
        
        temp_paramter_store=[]
        for counterLayer in range(numLayer):
            wbLayer=[finalWeightsList[counterLayer],finalBiasList[counterLayer]]
            temp_paramter_store.append(wbLayer)
            finalModel.layers[counterLayer].set_weights(wbLayer)
        parameter_store.append(temp_paramter_store)
        
            
        '''This part is for validation in distributed manner'''
        validation_silo_size=400
        validation_aucroc=[]
        validation_acupr=[]
        base_index=numNode*step
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
    
    
    
'''Get the trained model by FeARH'''
parameter=parameter_store[aucroc_store.index(max(aucroc_store))]


