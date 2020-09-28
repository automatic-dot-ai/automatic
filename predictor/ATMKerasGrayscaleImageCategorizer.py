
# @component {
#	"kind" : "predictor",
#	"language" : "py",
#	"description" : "Performs prediction using previous trained model",
#	"permissions": "public",
#	"properties": [
#		{ "name": "Pixel width" , "field": "width", "kind": "integer", "min": 8, "max": 1000, "required": true, "default": 28 },
#		{ "name": "Pixel height" , "field": "height", "kind": "integer", "min": 8, "max": 1000, "required": true, "default": 28 }
#	],
#	"inputs": ["data:img"],
#	"outputs": ["output:dict"],
#	"dependencies": ["tensorflow"],
#	"readme" : "",
#	"license" : ""
# }
from keras.models import load_model
from keras import backend as K  
from keras.models import load_model
from keras.preprocessing import image
import numpy as np

global_model = None

def ATMKerasGrayscaleImageCategorizer(ATM):
	#img = image.img_to_array(ATM.inputs["data"])
	img = np.array([ATM.inputs["data"]], dtype=np.float32)
	img_rows = ATM.props["height"]
	img_cols = ATM.props["width"]
	if K.image_data_format() == 'channels_first':
		img = img.reshape(1, 1, img_rows, img_cols)
	else:
		# default
		img = img.reshape(1, img_rows, img_cols, 1)

	img = img.astype('float32')
	img = img / 255.0
	global global_model
	if global_model == None:
		global_model = load_model(ATM.load("model"));
	category = global_model.predict_classes(img)
	ATM.report({ "name": "predictions", "predictions": category.tolist() })
	ATM.output({ 'results': category.tolist() });

