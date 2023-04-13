import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

base = pd.read_csv('characters.csv')
predictors = base.iloc[:,0:6].values
classes = base.iloc[:,6].values

def create_network():
    classifier = Sequential()
    classifier.add(Dense(units = 6, activation = 'relu', 
                         kernel_initializer = 'random_uniform', input_dim = 6))
    classifier.add(Dense(units = 6, activation = 'relu', 
                         kernel_initializer = 'random_uniform'))
    classifier.add(Dense(units = 1, activation = 'sigmoid'))

    optimizer = keras.optimizers.Adam(learning_rate = 0.001, decay=0.0001, clipvalue=0.5)
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['binary_accuracy'])
    
    return classifier

classifier = KerasClassifier(build_fn = create_network, epochs=100, batch_size=10)

results = cross_val_score(estimator = classifier, X=predictors, 
                          y=classes, cv=10, scoring='accuracy')

mean = results.mean()
deviation = results.std()