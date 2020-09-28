#
# @component {
#       "kind" : "trainer",
#       "language" : "py",
#       "description" : "Performs grid search over the 'hyper' parameter for a decision tree regressor trained on the input data",
#       "permissions": "public",
#       "properties": [
#               { "name": "Hyperparameter" , "field": "hyperparameter", "kind": "string", "minlen": 2, "maxlen": 32, "required": true, "default": "max_depth" },
#               { "name": "Min" , "field": "min", "kind": "number", "min": -100000, "max": 100000, "required": true, "default": 1 },
#               { "name": "Max" , "field": "max", "kind": "number", "min": -100000, "max": 100000, "required": true, "default": 11 }
#       ],
#       "inputs": ["X:pandas", "y:pandas"],
#       "outputs": ["X:pandas", "y:pandas"],
#       "dependencies": ["pandas", "sklearn"],
#       "readme" : "",
#       "license" : "",
#	"links": ["https://www.ritchieng.com/machine-learning-project-boston-home-prices/"]
# }
from sklearn.metrics import r2_score
from sklearn.metrics import make_scorer
from sklearn.tree import DecisionTreeRegressor
from sklearn.grid_search import GridSearchCV
import pandas as pd

def ATMSKLearnDecisionTreeRegressorGridSearch(ATM):
	X = ATM.inputs["X"]
	y = ATM.inputs["y"]
        cv_sets = ShuffleSplit(X.shape[0], n_iter = 10, test_size = 0.20, random_state = 0)
        regressor = DecisionTreeRegressor(random_state=0)
        params = dict
        params[ATM.props["hyperparameter"]]=range(ATM.props["min"], ATM.props["max"])
        scoring_fnc = make_scorer(performance_metric)
        grid = GridSearchCV(regressor, params, cv=cv_sets, scoring=scoring_fnc)
#grid_search = grid_search.fit(X_train, y_train, callbacks=[MyCallback()]) worked for me.

        grid = grid.fit(X, y)
	model = grid.best_estimator_
	ATM.report({ 'name': "stats", 'stats': model.get_params() })
	ATM.save("model", pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL))
	ATM.output({ 'X': X, 'y': y })

	""" Calculates and returns the performance score between 
		true and predicted values based on the metric chosen. """
	def performance_metric(y_true, y_predict):
		score = r2_score(y_true, y_predict)
		return score


