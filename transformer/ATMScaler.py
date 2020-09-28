
# @component {
#	"kind" : "transformer",
#	"language" : "py",
#	"description" : "Normalizes X values to be between 0 and 1",
#	"permissions": "public",
#	"inputs": ["X:pandas", "y:pandas"],
#	"outputs": ["X:pandas", "y:pandas"],
#	"dependencies": ["pandas", "sklearn"],
#	"readme" : "",
#	"license" : ""
# }
from sklearn import preprocessing
def ATMScaler(ATM):
	X = ATM.inputs['X']
	y = ATM.inputs['y']
	min_max_scaler = preprocessing.MinMaxScaler()
	X_scale = min_max_scaler.fit_transform(X)
	ATM.output({ 'X': X_scale, 'y': y })

