
# @component {
#	"kind" : "datasource",
#	"language" : "py",
#	"description" : "",
#	"permissions": "public",
#	"properties": [
#		{ "name": "Data URL" , "field": "data", "kind": "url", "minlen": 8, "maxlen": 256, "required": true },
#		{ "name": "Labels URL" , "field": "labels", "kind": "url", "minlen": 8, "maxlen": 256, "required": true }
#	],
#	"outputs": ["X:pandas", "y:pandas"],
#	"dependencies": ["pandas"],
#	"readme" : "",
#	"license" : ""
# }
import pandas as pd
def ATMCSVSupervisedDataSource(ATM):
	X = pd.read_csv(ATM.props["data"])
	y = pd.read_csv(ATM.props["labels"])
	ATM.report({ 'name': "stats", 'stats': { 'features': list(X.columns), 'target': list(y.columns)[0], 'rows': X.shape } })
	ATM.report("dataset has {0} data points with {1} variables each".format(*X.shape))
	ATM.output({ 'X': X, 'y': y })

