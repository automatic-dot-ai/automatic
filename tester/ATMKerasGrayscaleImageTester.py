#
# @component {
#       "kind" : "tester",
#       "language" : "py",
#	"description": "Tests a model",
#       "permissions": "public",
#       "properties": [
#       ],
#	"inputs": ["X:img[]", "y:string[]"],
#       "dependencies": ["pandas", "tensorflow"],
#       "readme" : "",
#       "license" : "",
#	"links": []
# }
from keras import backend as K
from keras.utils import to_categorical
from keras.models import load_model

def ATMKerasGrayscaleImageTester(ATM):
	img_rows = ATM.props.get("height") or 28
	img_cols = ATM.props.get("width") or 28
	X_test = ATM.inputs["X"]
	y_test = ATM.inputs["y"]
	if K.image_data_format() == 'channels_first':
		# Theano
		X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
		input_shape = (1, img_rows, img_cols)
	else:
		X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
		input_shape = (img_rows, img_cols, 1)

	y_test = to_categorical(y_test)
	test = X_test.astype('float32')
	X_test = test / 255.0
	model = load_model(ATM.load("model"))
	acc = model.evaluate(X_test, y_test)[1]
	ATM.report({ 'name': "stats", 'stats': { 'accuracy': acc } })


