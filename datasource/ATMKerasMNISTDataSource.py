
# @component {
#	"kind" : "datasource",
#	"language" : "py",
#	"description" : "",
#	"permissions": "public",
#	"properties": [
#		{ "name": "Test dataset" , "field": "test", "kind": "boolean", "required": false, "hint": "Set to true if want to use test dataset" }
#	],
#	"outputs": ["X:img[]", "y:onehot"],
#	"dependencies": ["tensorflow"],
#	"readme" : "",
#	"license" : ""
# }
import tensorflow as tf
from keras.datasets import mnist
def ATMKerasMNISTDataSource(ATM):
	(X_train, y_train), (X_test, y_test) = mnist.load_data()
	if ATM.props.get("test"):
		X = X_test
		y = y_test
	else:
		X = X_train
		y = y_train
	ATM.output({ 'X': X, 'y': y })

