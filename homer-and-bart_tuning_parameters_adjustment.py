import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

base = pd.read_csv('characters.csv')
predictors = base.iloc[:,0:6].values
classes = base.iloc[:,6].values

classes = (classes == 'Homer')

def create_network(optimizer, loos, kernel_initializer, activation, neurons):
    classifier = Sequential()
    classifier.add(Dense(units = neurons, activation = activation, 
                         kernel_initializer = kernel_initializer, input_dim = 6))
    classifier.add(Dropout(0.2))
    classifier.add(Dense(units = neurons, activation = activation,
                         kernel_initializer = kernel_initializer))
    classifier.add(Dropout(0.2))
    classifier.add(Dense(units = 1, activation = 'sigmoid'))

    # optimizer = keras.optimizers.Adam(learning_rate = 0.001, decay=0.0001, clipvalue=0.5)
    
    classifier.compile(optimizer = optimizer, loss = loos, metrics = ['binary_accuracy'])
    
    return classifier

classifier = KerasClassifier(build_fn = create_network, epochs = 100, batch_size = 10)

parameters = {
    'batch_size': [10, 30],  
    'epochs': [50, 100],
    'optimizer': ['adam', 'sgd'],
    'loos': ['binary_crossentropy', 'hinge'],
    'kernel_initializer': ['random_uniform', 'normal'],
    'activation': ['relu', 'tanh'],
    'neurons': [12, 6]
}

grid_search = GridSearchCV(estimator = classifier, 
                           param_grid = parameters, 
                           scoring = 'accuracy', 
                           cv = 5)

grid_search = grid_search.fit(predictors, classes)
best_parameters = grid_search.best_params_
best_precision = grid_search.best_score_