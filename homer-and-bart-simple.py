import pandas as pd
import keras
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from keras.models import Sequential
from keras.layers import Dense

base = pd.read_csv('characters.csv')
predictors = base.iloc[:,0:6].values
classes = base.iloc[:,6].values

classes = (classes == 'Homer')

predictors_training, predictors_test, classes_training, classes_test = train_test_split(
    predictors, classes, test_size=0.25)

classifier = Sequential()
classifier.add(Dense(units = 6, activation = 'relu', 
                     kernel_initializer = 'random_uniform', input_dim = 6))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['binary_accuracy'])

classifier.fit(predictors_training, classes_training, batch_size=10, epochs=100)

predictions = classifier.predict(test_predictors)
predictions = (predictions > 0.5)


precision = accuracy_score(test_class, predictions)
matriz = confusion_matrix(test_class, predictions)

result = classifier.evaluate(test_predictors, test_class)

