
# @component {
#	"kind" : "augmentor",
#	"language" : "py",
#	"description" : "",
#	"permissions": "public",
#	"properties": [
#		{ "name": "Fuzziness" , "field": "fuzziness", "kind": "number", "min": 0, "max": 1.0, "default": 0.1 },
#		{ "name": "Augmentation factor" , "field": "factor", "kind": "number", "min": 0, "max": 10, "default": 1 },
#		{ "name": "Column", "field": "column", "kind": "string", "required": true, "feature": true }
#	],
#	"inputs": ["X:pandas", "y:pandas"],
#	"outputs": ["X:pandas", "y:pandas"],
#	"dependencies": ["pandas"],
#	"readme" : "",
#	"license" : ""
# }
import pandas as pd
import random
def ATMCSVFuzzyValueAugmentor(ATM):
	X = ATM.inputs["X"]
	y = ATM.inputs["y"]
	data = pd.concat([X, y], axis=1, sort=False)
	target_column = ATM.props["column"]
	fuzziness = ATM.props["fuzziness"]
	factor = ATM.props["factor"]
	temp_df = []
	colindex = data.columns.get_loc(target_column)
	for row in data.itertuples(index=False):
		if random.random() < factor:
			r = list(row)
			r[colindex] = r[colindex] + random.random() * 2 * fuzziness - fuzziness
			temp_df.extend([r]*max(1, int(factor)))
		temp_df.append(list(row))
	data = pd.DataFrame(temp_df, columns=data.columns)
	X = data.iloc[:,:-1]
	y = data.iloc[:,-1].to_frame()
	ATM.report("dataset now has {0} data points with {1} variables each".format(*X.shape))
	ATM.report("target now has {0} data points with {1} variables each".format(*y.shape))
	ATM.output({ 'X': X, 'y': y })

