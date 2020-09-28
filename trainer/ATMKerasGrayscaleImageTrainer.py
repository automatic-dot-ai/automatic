
# @component {
#	"kind" : "trainer",
#	"language" : "py",
#	"description" : "Train model to recognize categories of grayscale images (MNIST)",
#	"permissions": "public",
#	"properties": [
#		{ "name": "Pixel width" , "field": "width", "kind": "integer", "min": 8, "max": 1000, "required": true, "default": 28 },
#		{ "name": "Pixel height" , "field": "height", "kind": "integer", "min": 8, "max": 1000, "required": true, "default": 28 },
#		{ "name": "Epochs" , "field": "epochs", "kind": "integer", "min": 1, "max": 1000, "required": true, "default": 4, "hint": "Tensorflow recommends 10 which takes too long at about 20 minutes using speed 3 automatic" },
#		{ "name": "Batch size" , "field": "batch_size", "kind": "integer", "min": 2, "max": 1000, "required": true, "default": 32 }
#	],
#	"inputs": ["X:img[]", "y:string[]"],
#	"outputs": ["X:img[]", "y:string[]"],
#	"dependencies": ["tensorflow", "numpy"],
#	"readme" : "",
#	"license" : "",
#	"links" : ["https://machinelearningmastery.com/how-to-develop-a-convolutional-neural-network-from-scratch-for-mnist-handwritten-digit-classification/", "https://towardsdatascience.com/a-simple-2d-cnn-for-mnist-digit-recognition-a998dbc1e79a", "https://keras.io/examples/mnist_cnn/" ]
# }
import tensorflow as tf
from tensorflow import keras
from numpy import mean
from numpy import std
from keras.datasets import mnist
from keras import backend as K  
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD
from keras.utils import to_categorical

from keras import callbacks

def ATMKerasGrayscaleImageTrainer(ATM):
	img_rows = ATM.props.get("height") or 28
	img_cols = ATM.props.get("width") or 28
	X_train = ATM.inputs["X"]
	y_train = ATM.inputs["y"]
	if K.image_data_format() == 'channels_first':
		# Theano
		X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
		input_shape = (1, img_rows, img_cols)
	else:
		X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
		input_shape = (img_rows, img_cols, 1)

	y_train = to_categorical(y_train)
	train_norm = X_train.astype('float32')
	X_train = train_norm / 255.0

	def atm_progress_callback(framework, purpose):
		if framework == "keras":
			return callbacks.LambdaCallback(
				on_epoch_end=lambda epoch, logs: 
					ATM.report({ "name": "progress", 'purpose': purpose, 'progress': (epoch + 1) / ATM.props["epochs"], 'loss': logs['loss'], 'finished': False, 'gpu' : tf.config.experimental.list_physical_devices('GPU') }),
				on_train_end=lambda logs: 
					ATM.report({ "name": "progress", 'purpose': purpose, 'finished': True, 'gpu' : tf.config.experimental.list_physical_devices('GPU')  })
			)
		return None;

	def define_model():
		model = Sequential()
		model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=input_shape))
		model.add(MaxPooling2D((2, 2)))
		model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
		model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
		model.add(MaxPooling2D((2, 2)))
		model.add(Flatten())
		model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
		model.add(Dense(10, activation='softmax'))
		opt = SGD(lr=0.01, momentum=0.9)
		model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
		return model

	model = define_model()
	model.fit(X_train, y_train, epochs=ATM.props["epochs"], batch_size=ATM.props["batch_size"], verbose=0, 
		callbacks=[atm_progress_callback("keras", "train")])
	filename = ATM.getTmpFilename("model")
	model.save(filename)
	model_h5 = open(filename, 'rb').read()
	ATM.save("model", model_h5)
	ATM.output(ATM.inputs);

