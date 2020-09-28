
# @component {
#	"kind" : "filter",
#	"language" : "py",
#	"description" : "",
#	"permissions": "public",
#	"properties": [
#		{ "name": "Remove NaN values" , "field": "remove_nan_values", "kind": "boolean", "default": true },
#		{ "name": "Remove empty values" , "field": "remove_empty_values", "kind": "boolean", "default": true }
#	],
#	"inputs": ["X:pandas", "y:pandas"],
#	"outputs": ["X:pandas", "y:pandas"],
#	"dependencies": ["pandas", "numpy"],
#	"readme" : "",
#	"license" : ""
# }
import pandas as pd
import numpy as np
def ATMCSVMissingColumnFilter(ATM):
	X = ATM.inputs.get("X")
	y = ATM.inputs.get("y")
	if (X is None) | X.empty | (y is None) | y.empty:
		return ATM.report({ 'name': "log", 'level': "error", 'message': "Inputs X or y not found", 'at': "ATMCSVMissingColumnFilter" })
	data = pd.concat([X, y], axis=1, sort=False)
	if ATM.props.get("remove_empty_values"):
		data = data.replace('', np.nan).dropna(how='any',axis=0)
	elif ATM.props.get("remove_nan_values"):
		data = data.dropna(how='any',axis=0) 
	ATM.report("dataset has {0} data points with {1} variables each".format(*X.shape))
	ATM.report("target has {0} data points with {1} variables each".format(*y.shape))
	X = data.iloc[:,:-1]
	y = data.iloc[:,-1].to_frame()
	ATM.report("dataset now has {0} data points with {1} variables each".format(*X.shape))
	ATM.report("target now has {0} data points with {1} variables each".format(*y.shape))
	ATM.output({ 'X': X, 'y': y })
	
