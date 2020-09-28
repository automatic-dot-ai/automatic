
# 
# @component {
#	"kind": "tester",
#	"language": "py",
#	"description": "Tests a model",
#	"permissions": "public",
#	"inputs": ["X:pandas", "y:pandas"],
#	"outputs": ["X:pandas", "y:pandas"],
#	"dependencies": ["pandas", "sklearn", "numpy"],
#	"readme": "",
#	"license": ""
# }
import pickle
import numpy as np
from sklearn.metrics import mean_squared_error
def ATMSKLearnModelTester(ATM):
	X = ATM.inputs["X"]
	y = ATM.inputs["y"]
	print("filename to load model from=", ATM.load("model"))
	with open(ATM.load("model"), 'rb') as filehandle:
		model = pickle.load(filehandle)
	y_predicted = model.predict(X)
	rmse = np.sqrt(mean_squared_error(y, y_predicted))
	r2 = round(model.score(X, y),2)
	ATM.report({ 'name': "stats", 'stats': { "RMSE": rmse, "R^2": r2 } })
	ATM.output({ 'X': X, 'y': y, 'y_predicted': y_predicted })


