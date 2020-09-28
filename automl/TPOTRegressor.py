
# @component {
#	"kind" : "automl",
#	"language" : "py",
#	"description" : "Use genetic algorithms to determine best model algorithm",
#	"permissions": "public",
#	"properties": [
#		{ "name": "Generations" , "field": "generations", "kind": "integer", "min": 2, "max": 100, "required": true, "default": 5 },
#		{ "name": "Population size" , "field": "population_size", "kind": "integer", "min": 2, "max": 100, "required": true, "default": 50 },
#		{ "name": "verbosity" , "field": "verbosity", "kind": "integer", "max": 2, "min": 100, "required": true, "default": 2 },
#		{ "name": "random_state" , "field": "random_state", "kind": "integer", "min": 2, "max": 100, "required": true, "default": 42 }
#	],
#	"inputs": ["X:pandas", "y:pandas", "X_test:pandas", "y_test:pandas"],
#	"outputs": ["X:pandas", "y:pandas"],
#	"dependencies": ["tpot", "sklearn"],
#	"readme" : "",
#	"license" : "",
#	"links" : ["https://github.com/EpistasisLab/tpot"]
# }
from tpot import TPOTRegressor

# https://github.com/EpistasisLab/tpot
def TPOTRegressor(ATM):
	X = ATM.inputs["X"]
	y = ATM.inputs["y"]
	tpot = TPOTRegressor(generations=ATM.props["generations"], population_size=ATM.props["population_size"], verbosity=ATM.props["verbosity"], random_state=ATM.props["random_state"])
	tpot.fit(X, y)
	ATM.report({ 'name': "stats", 'stats': { 'score': tpot.score(payload.X_test, y_test) } })
	ATM.report({ 'name': "log", 'payload': { 'model': tpot.export() } });
	ATM.save("model.tpot", tpot.export())

