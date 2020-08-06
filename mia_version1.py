"""

This one implemented FeARH

"""



import numpy as np

from absl import app
from absl import flags
import pandas as pd

import tensorflow as tf
from tensorflow.keras import layers

from sklearn.model_selection import train_test_split
import keras
import keras.backend as K
from keras import regularizers
import NN_GA as ga
from mia.estimators import ShadowModelBundle, AttackModelBundle, prepare_attack_data
from sklearn.metrics import roc_auc_score,average_precision_score


NUM_CLASSES = 1
#WIDTH = 32
#HEIGHT = 32
#CHANNELS = 3
SHADOW_DATASET_SIZE = 2500
ATTACK_TEST_DATASET_SIZE = 5000

scala=2600


FLAGS = flags.FLAGS
flags.DEFINE_integer(
    "target_epochs", 12, "Number of epochs to train target and shadow models."
)
flags.DEFINE_integer("attack_epochs", 12, "Number of epochs to train attack models.")
flags.DEFINE_integer("num_shadows", 3, "Number of epochs to train attack models.")



def target_model_fn():
    """The architecture of the target (victim) model.
    The attack is white-box, hence the attacker is assumed to know this architecture too."""

    model = tf.keras.models.Sequential()

    model.add(layers.Dense(4, activation="relu", input_dim=2913, kernel_regularizer=regularizers.l2(0.01)))
    model.add(layers.Dense(2, activation="relu", kernel_regularizer=regularizers.l2(0.01)))
    model.add(layers.Dense(1, activation="sigmoid"))
    model.compile(optimizer='Nadam', loss='binary_crossentropy')
    return model


def attack_model_fn():
    """Attack model that takes target model predictions and predicts membership.
    Following the original paper, this attack model is specific to the class of the input.
    AttachModelBundle creates multiple instances of this model for each class.
    """
    model = tf.keras.models.Sequential()

    model.add(layers.Dense(128, activation="relu", input_shape=(NUM_CLASSES,)))

    model.add(layers.Dropout(0.3, noise_shape=None, seed=None))
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dropout(0.2, noise_shape=None, seed=None))
    model.add(layers.Dense(64, activation="relu"))

    model.add(layers.Dense(1, activation="sigmoid"))
    model.compile("adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def train_target(X,Y):
    '''the number of node in each layer in neural network '''
    numUnitLayer0=4
    numUnitLayer1=2
    numUnitLayer2=1
    numLayer=3
    numUnitLayer=[numUnitLayer0,numUnitLayer1,numUnitLayer2]#improve convenience for later iteration
    
    
#    numFeature=X.shape[1]#The total number of madicine kind
    xFinalTest=X[25000:]#####################################################
#    y_tes/t=Y[25000:]#########################################################
    
    
    numModel=8##
    numNode=8
    step=scala##(the number of data pieces in each sub dataset)changable

    exchangeRate=0.4
    
    weightsGlobalList,biasGlobalList=ga.initParameter2(numUnitLayer0,numUnitLayer1,numUnitLayer2,
                                                               X,numModel,numUnitLayer,numLayer)
    sumDataPiece=np.zeros((numModel))
    
    
    stop_counter=0
    num_hybrid=0
    
    current_aucroc_score=0.01
    previous_aucroc_score=0.01
    
    halting_condition=0.0001
    aucroc_store=[]
    parameter_store=[]
    
#    start=time.time()
    
    
    while stop_counter<3:
#        print(time.time()-start)
        num_hybrid+=1
        datasetIndex=[0,1,2,3,4,5,6,7,8]
        '''Here we start a hybridization cycle of interation over all eight nodes'''
        for counterNode in range(numNode):
            counterDatasetIndex=datasetIndex[counterNode]
            xLocalTrain,xLocalTest,yLocalTrain,yLocalTest=ga.selectDataset(datasetIndex[counterDatasetIndex],
                                                                                   X,Y,step)
            
            sumDataPiece[counterDatasetIndex]=sumDataPiece[counterDatasetIndex]+xLocalTrain.shape[0]+xLocalTest.shape[0]
            
            K.clear_session()
            
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
        
    
    
    
    '''Build model for testing'''
    parameter_index=aucroc_store.index(max(aucroc_store))
    test_parameter=parameter_store[parameter_index]
    test_model=ga.buildModelList(numUnitLayer0,numUnitLayer1,numUnitLayer2,xFinalTest,numModel)
    for counterLayer in range(numLayer):
        test_model.layers[counterLayer].set_weights(test_parameter[counterLayer])
    return test_model



def demo(argv):
    x=pd.read_csv(r"C:\document\education\python\python\predict mortality\data\drug_table_mortality_with_no.csv")
    y=pd.read_csv(r"C:\document\education\python\python\predict mortality\data\mortality_table.csv")

#    (X_train, y_train), (X_test, y_test) = get_data()
    X,Y=ga.shuffleData(x,y)
    X_test=X[25000:]#####################################################
    y_test=Y[25000:]#########################################################
    X_train=X[:8*scala]
    y_train=Y[:8*scala]

    # Train the target model.
    print("Training the target model...")
    target_model = train_target(X,Y)

    # Train the shadow models.
    smb = ShadowModelBundle(
        target_model_fn,
        shadow_dataset_size=SHADOW_DATASET_SIZE,
        num_models=FLAGS.num_shadows,
    )

    # We assume that attacker's data were not seen in target's training.
    attacker_X_train, attacker_X_test, attacker_y_train, attacker_y_test = train_test_split(
        X_test, y_test, test_size=0.1
    )
    print(attacker_X_train.shape, attacker_X_test.shape)

    print("Training the shadow models...")
    X_shadow, y_shadow = smb.fit_transform(
        attacker_X_train,
        attacker_y_train,
        fit_kwargs=dict(
            epochs=FLAGS.target_epochs,
            verbose=True,
            validation_data=(attacker_X_test, attacker_y_test),
        ),
    )
    print("shadow_shape")
    print(X_shadow.shape)
    print(y_shadow)
    # ShadowModelBundle returns data in the format suitable for the AttackModelBundle.
    amb = AttackModelBundle(attack_model_fn, num_classes=NUM_CLASSES)

    # Fit the attack models.
    print("Training the attack models...")
    amb.fit(
        X_shadow, y_shadow, fit_kwargs=dict(epochs=FLAGS.attack_epochs, verbose=True)
    )

    # Test the success of the attack.

    # Prepare examples that were in the training, and out of the training.
    data_in = X_train[:ATTACK_TEST_DATASET_SIZE], y_train[:ATTACK_TEST_DATASET_SIZE]
    data_out = X_test[:ATTACK_TEST_DATASET_SIZE], y_test[:ATTACK_TEST_DATASET_SIZE]

    # Compile them into the expected format for the AttackModelBundle.
    attack_test_data, real_membership_labels = prepare_attack_data(
        target_model, data_in, data_out
    )

    # Compute the attack accuracy.
    attack_guesses = amb.predict(attack_test_data)
    attack_accuracy = np.mean(attack_guesses == real_membership_labels)

    print(attack_accuracy)


if __name__ == "__main__":
    app.run(demo)
    
    
"""

0.5
0.4984



"""