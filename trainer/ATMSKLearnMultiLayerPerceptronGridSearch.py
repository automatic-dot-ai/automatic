#
# @component {
#       "kind" : "trainer",
#       "language" : "py",
#       "description" : "Performs grid search over several 'hyper' parameters for a multi-layer regressor trained on the input data",
#       "permissions": "public",
#       "inputs": ["X:pandas", "y:pandas"],
#       "outputs": ["X:pandas", "y:pandas"],
#       "dependencies": ["pandas", "sklearn"],
#       "readme" : "",
#       "license" : "",
#	"links": ["https://dataoutpost.wordpress.com/2018/05/28/housing-price-prediction-with-multi-layer-perceptron/"]
# }
from sklearn.metrics import r2_score
from sklearn.metrics import make_scorer
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

def ATMSKLearnMultiLayerPerceptronGridSearch(ATM):
	X = ATM.inputs["X"]
	y = ATM.inputs["y"]
	X_train, X_test, y_train, y_test = train_test_split(X, y)
	scaler = StandardScaler()
	scaler.fit(X_train)
	X_train = scaler.transform(X_train)
	X_test = scaler.transform(X_test)
 
	tuned_parameters = [{'hidden_layer_sizes': [1,2,3,4,5,6,7,8,9,10,20,30,40], 'activation': ['relu'], 'solver':['lbfgs'], 'alpha':[0.0001], 'batch_size':['auto'], 'learning_rate':['constant'], 'learning_rate_init':[0.001], 'max_iter':[500]}]
	grid = GridSearchCV(MLPRegressor(), tuned_parameters, cv=5)
	grid.fit(X_train, y_train)
 
	#mlp = MLPRegressor(hidden_layer_sizes=(20,), activation='logistic', solver='lbfgs', alpha=0.0001, batch_size='auto', learning_rate='constant', learning_rate_init=0.001, max_iter=500)
	#mlp.fit(X_train,y_train)
	train_mse = mean_squared_error(y_train, grid.predict(X_train))
	test_mse = mean_squared_error(y_test, grid.predict(X_test))
 
	print(grid.best_params_)
	print(grid.best_score_)

	model = grid.best_estimator_
	params = model.get_params()
	params["name"] = "stats"
	params["stats"] = { 'train_mse': train_mse, 'test_mse': test_mse }
	ATM.report(params)
	ATM.set("model", model)
	ATM.output({ 'X': X, 'y': y })


