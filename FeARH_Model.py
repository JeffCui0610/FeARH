'''
membership inference attack is conducted on FeARH

'''
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import keras.backend as K
from keras import regularizers
import FeARH_lib as ga
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score


def train_target(X, Y):
    '''the number of node in each layer in neural network '''
    numUnitLayer0 = 4
    numUnitLayer1 = 2
    numUnitLayer2 = 1
    numLayer = 3
    numUnitLayer = [numUnitLayer0, numUnitLayer1, numUnitLayer2]  # improve convenience for later iteration

    xFinalTest = X[25000:]  #####################################################

    numModel = 8  ##
    numNode = 8
    step = scala  ##(the number of data pieces in each sub dataset)changable

    exchangeRate = 0.5

    weightsGlobalList, biasGlobalList = ga.initParameter2(numUnitLayer0, numUnitLayer1, numUnitLayer2,
                                                          X, numModel, numUnitLayer, numLayer)
    sumDataPiece = np.zeros((numModel))

    first_parameter_store = []

    stop_counter = 0
    num_hybrid = 0

    current_aucroc_score = 0.01
    previous_aucroc_score = 0.01

    halting_condition = 0.0001
    aucroc_store = []
    parameter_store = []

    while stop_counter < 3:
        num_hybrid += 1
        datasetIndex = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        '''Here we start a hybridization cycle of interation over all eight nodes'''
        for counterNode in range(numNode):
            counterDatasetIndex = datasetIndex[counterNode]
            xLocalTrain, xLocalTest, yLocalTrain, yLocalTest = ga.selectDataset(datasetIndex[counterDatasetIndex],
                                                                                X, Y, step)

            sumDataPiece[counterDatasetIndex] = sumDataPiece[counterDatasetIndex] + xLocalTrain.shape[0] + \
                                                xLocalTest.shape[0]

            K.clear_session()

            nn = ga.buildModelList(numUnitLayer0, numUnitLayer1, numUnitLayer2, X, 1)
            '''Load parameters in local node'''
            ga.loadNNModel(numModel, nn, weightsGlobalList, biasGlobalList, counterNode, numLayer)
            '''Using Gradient descent to train this model'''
            nn.compile(optimizer='Nadam', loss='binary_crossentropy')
            nn.fit(xLocalTrain, yLocalTrain, nb_epoch=10, verbose=0)

            '''retrieve trained parameters back to matrics'''
            for counterLayer in range(numLayer):
                weightsGlobalList[counterLayer][counterNode] = nn.layers[counterLayer].get_weights()[0]
                biasGlobalList[counterLayer][counterNode] = nn.layers[counterLayer].get_weights()[1]
            '''Hybrid among all nodes in pairs'''
        weightsGlobalList, biasGlobalList = ga.hybrid2(numUnitLayer, xLocalTrain, numModel,
                                                       exchangeRate, weightsGlobalList, biasGlobalList)

        '''Retrieve the first model's parameter'''
        #        first_model = ga.buildModelList(numUnitLayer0,numUnitLayer1,numUnitLayer2,xFinalTest,numModel)
        temp_first_parameter_store = []
        for counterLayer in range(numLayer):
            wbLayer1 = [weightsGlobalList[counterLayer][0], biasGlobalList[counterLayer][0]]
            temp_first_parameter_store.append(wbLayer1)
        first_parameter_store.append(temp_first_parameter_store)

        finalWeightsList, finalBiasList = ga.averageParameter2(sumDataPiece, X, numUnitLayer,
                                                               weightsGlobalList, biasGlobalList, numModel, numLayer)
        finalModel = ga.buildModelList(numUnitLayer0, numUnitLayer1, numUnitLayer2, xFinalTest, numModel)

        temp_paramter_store = []
        for counterLayer in range(numLayer):
            wbLayer = [finalWeightsList[counterLayer], finalBiasList[counterLayer]]
            temp_paramter_store.append(wbLayer)
            finalModel.layers[counterLayer].set_weights(wbLayer)
        parameter_store.append(temp_paramter_store)

        '''This part is for validation in distributed manner'''
        validation_silo_size = 250
        validation_aucroc = []
        validation_acupr = []
        base_index = numNode * step
        for i in range(numNode):
            validation_x = X[base_index + (i * validation_silo_size):base_index + ((i + 1) * validation_silo_size)]  #
            validation_y = Y[base_index + (i * validation_silo_size):base_index + ((i + 1) * validation_silo_size)]
            pred_validation = finalModel.predict(validation_x)
            aucrocScore_validation = roc_auc_score(validation_y, pred_validation)
            aucprScore_validation = average_precision_score(validation_y, pred_validation)
            validation_aucroc.append(aucrocScore_validation)
            validation_acupr.append(aucprScore_validation)
        current_aucroc_score = np.mean(validation_aucroc)

        if (current_aucroc_score - previous_aucroc_score) / previous_aucroc_score <= halting_condition:
            stop_counter += 1
        else:
            stop_counter = 0
        aucroc_store.append(current_aucroc_score)
        previous_aucroc_score = max(aucroc_store)

    parameter_index = aucroc_store.index(max(aucroc_store))
    test_parameter1 = first_parameter_store[parameter_index]
    test_model1 = ga.buildModelList(numUnitLayer0, numUnitLayer1, numUnitLayer2, xFinalTest, numModel)
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


x = pd.read_csv(r"...\drug_table_mortality_with_no.csv")
y = pd.read_csv(r"...\mortality_table.csv")
scala = 2000 # Number of records for each node, changeable
X, Y = ga.shuffleData(x, y)
target_model = train_target(X, Y) # Return the model of FeARH
