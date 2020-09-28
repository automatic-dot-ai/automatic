
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
#	"dependencies": ["pytorch", "numpy"],
#	"readme" : "",
#	"license" : ""
# }

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

global_model = None

def ATMTorchGrayscaleImageCategorizer(ATM):
	#img = image.img_to_array(ATM.inputs["data"])
	img = np.array([ATM.inputs["data"]], dtype=np.float32)
	img_rows = ATM.props["height"]
	img_cols = ATM.props["width"]
	#img = img.reshape(1, img_rows, img_cols, 1)
	img = img.reshape(1, 1, img_rows, img_cols)
	img = img.astype('float32')
	img = img / 255.0
	img  = torch.from_numpy(img) ## .float()

	global global_model
	if global_model == None:
		state_dict = torch.load(ATM.load("model", True))
		if state_dict is None:
			return ATM.report({ 'name': "log", 'level': "error", 'message': "Model at '" + "model" + "' not found", 'at': "ATMTorchGrayscaleImageCategorizer" })
		global_model = Net();
		global_model.load_state_dict(state_dict)

		use_cuda = ATM.props.get("cuda") and torch.cuda.is_available()
		device = torch.device("cuda" if use_cuda else "cpu")
		global_model = global_model.to(device)
	# Turn off gradients to speed up this part
	with torch.no_grad():
		category = global_model(img).numpy()[0].argmax()
	ATM.report({ "name": "predictions", "predictions": int(category) })
	ATM.output({ 'results': int(category) });

class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.conv1 = nn.Conv2d(1, 32, 3, 1)
		self.conv2 = nn.Conv2d(32, 64, 3, 1)
		self.dropout1 = nn.Dropout2d(0.25)
		self.dropout2 = nn.Dropout2d(0.5)
		self.fc1 = nn.Linear(9216, 128)
		self.fc2 = nn.Linear(128, 10)

	def forward(self, x):
		x = self.conv1(x)
		x = F.relu(x)
		x = self.conv2(x)
		x = F.relu(x)
		x = F.max_pool2d(x, 2)
		x = self.dropout1(x)
		x = torch.flatten(x, 1)
		x = self.fc1(x)
		x = F.relu(x)
		x = self.dropout2(x)
		x = self.fc2(x)
		output = F.log_softmax(x, dim=1)
		return output


