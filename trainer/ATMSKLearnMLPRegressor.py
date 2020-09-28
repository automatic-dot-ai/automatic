#
# @component {
#       "kind" : "trainer",
#       "language" : "py",
#       "description" : "Performs grid search over the 'hyper' parameter for a decision tree regressor trained on the input data",
#       "permissions": "public",
#       "properties": [
#               { "name": "Hidden layer sizes" , "field": "hidden_layer_sizes", "kind": "number", "min": 1, "max": 100, "required": true, "default": 20 },
#               { "name": "Activation" , "field": "activation", "kind": "menu", "choices": ["identity", "logistic", "tanh", "relu"], "required": true, "default": "relu" },
#               { "name": "Solver" , "field": "solver", "kind": "menu", "choices": ["sgd", "adam", "lbfgs"], "required": true, "default": "adam" },
#               { "name": "Alpha" , "field": "alpha", "kind": "number", "min": 0, "max": 1, "required": true, "default": 0.0001 },
#               { "name": "Batch size" , "field": "batch_size", "kind": "number", "min": 5, "max": 1000, "required": true, "default": 100 },
#               { "name": "Learning rate" , "field": "learning_rate", "kind": "menu", "choices": ["constant", "invscaling", "adaptive"], "required": true, "default": "constant" },
#               { "name": "Learning rate init" , "field": "learning_rate_init", "kind": "number", "min": 0.000001, "max": 1, "required": true, "default": 0.001 }
#       ],
#       "inputs": ["X:pandas", "y:pandas"],
#       "outputs": ["X:pandas", "y:pandas"],
#       "dependencies": ["pandas", "sklearn", "numpy"],
#       "readme" : "",
#       "license" : ""
# }
from keras.models import Sequential
from keras.layers import Dense
from keras.models import Sequential
from keras.layers import Dropout
from keras import regularizers
import pandas as pd
import numpy as np

def ATMSKLearnMLPRegressor(ATM):
	X = ATM.inputs["X"]
	y = ATM.inputs["y"]

	model = MLPRegressor(hidden_layer_sizes=(ATM.props["hidden_layer_sizes"],), activation=ATM.props["activation"], solver=ATM.props["solver"], alpha=ATM.props["alpha"], batch_size=ATM.props["batch_size"], learning_rate=ATM.props["learning_rate"], learning_rate_init=ATM.props["learning_rate_init"])
	model.fit(X,y)
	train_rmse = np.sqrt(mean_squared_error(y, rgr.predict(X)))

	ATM.report({ 'name': "stats", 'stats': { 'RMSE': train_rmse } })
	ATM.save("model", model)
	ATM.output({ 'X': X, 'y': y })


