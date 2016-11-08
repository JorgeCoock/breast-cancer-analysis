from keras.models import Sequential
from keras.layers import Dense
import numpy

#random seed
seed = 27
numpy.random.seed(seed)

dataset = numpy.loadtxt("breast-cancer-wisconsin.csv", delimiter=",")

X = dataset[:,1:11]

Y = dataset[:,11]

model = Sequential()

model.add(Dense(20, input_dim=10, init='uniform', activation='relu'))
model.add(Dense(12, init='uniform', activation='relu'))
model.add(Dense(6, init='uniform', activation='relu'))
model.add(Dense(1, init='uniform', activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X, Y, nb_epoch=300, batch_size=10)
