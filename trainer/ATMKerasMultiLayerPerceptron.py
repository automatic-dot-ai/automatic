#
# @component {
#       "kind" : "trainer",
#       "language" : "py",
#       "description" : "Performs grid search over the 'hyper' parameter for a decision tree regressor trained on the input data",
#       "permissions": "public",
#       "properties": [
#       ],
#       "inputs": ["X:pandas", "y:pandas"],
#       "outputs": ["X:pandas", "y:pandas"],
#       "dependencies": ["pandas", "sklearn", "tensorflow"],
#       "readme" : "",
#       "license" : ""
# }
from keras.models import Sequential
from keras.layers import Dense
from keras.models import Sequential
from keras.layers import Dropout
from keras import regularizers
import pandas as pd
from sklearn.model_selection import train_test_split


def ATMKerasMultiLayerPerceptron(ATM):
	X = ATM.inputs["X"]
	y = ATM.inputs["y"]

	X_train, X_val_and_test, Y_train, Y_val_and_test = train_test_split(X, y, test_size=0.3)
	X_val, X_test, Y_val, Y_test = train_test_split(X_val_and_test, Y_val_and_test, test_size=0.5)
	#layers = ATM.build("layers", X.shape)
	#if layers == None:
	layers = [
			Dense(1000, activation='relu', kernel_regularizer=regularizers.l2(0.01), input_shape=(X.shape[1],)),
			Dropout(0.3),
			Dense(1000, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
			Dropout(0.3),
			Dense(1000, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
			Dropout(0.3),
			Dense(1000, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
			Dropout(0.3),
			Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(0.01)),
		]
	model = Sequential(layers)
	model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
	hist = model.fit(X_train, Y_train, batch_size=32, epochs=100, validation_data=(X_val, Y_val))
	acc = model.evaluate(X_test, Y_test)[1]
	ATM.report({ 'name': "stats", 'stats': { 'accuracy': acc } })
	model.save("model.h5")
	model_h5 = open('model.h5', 'r')
	ATM.save("model", model_h5)
	ATM.output({ 'X': X, 'y': y, 'history': hist })


