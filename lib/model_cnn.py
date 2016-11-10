import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils

#random seed
seed = 7
numpy.random.seed(seed)

dataset = numpy.loadtxt("breast-cancer-wisconsin.csv", delimiter=",")
x_train = dataset[:,1:11]
y_train = dataset[:,10]

dataset_test = numpy.loadtxt("breast-cancer-wisconsin-test.csv", delimiter=",")
x_test = dataset_test[:,1:11]
y_test = dataset_test[:,10]

for index, line in enumerate(y_train):
    if line == 2.0:
        line = 0
    if line == 4.0:
        line = 1
    y_train[index] = line

for index, line in enumerate(y_test):
    if line == 2.0:
        line = 0
    if line == 4.0:
        line = 1
    y_test[index] = line

#normlize inputs from 1-10 to 0-1
x_train = x_train/10
x_test = x_test/10

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]
num_inputs = 10

def baseline_model():
# create model
	model = Sequential()
	model.add(Convolution2D(32, 5, 5, input_shape=(1, 28, 28), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.2))
	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dense(num_inputs, activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

# build the model
model = baseline_model()
# Fit the model
model.fit(x_train, y_train, validation_data=(x_test, y_test), nb_epoch=10, batch_size=200, verbose=2)
# Final evaluation of the model
scores = model.evaluate(x_test, y_test, verbose=0)
print("CNN Error: %.2f%%" % (100-scores[1]*100))
