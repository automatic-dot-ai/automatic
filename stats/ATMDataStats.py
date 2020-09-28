
#
# @component {
#       "kind" : "stats",
#       "language" : "py",
#       "description" : "",
#       "permissions": "public",
#       "properties": [
#       ],
#       "inputs": ["X:pandas", "y:pandas"],
#       "outputs": ["X:pandas", "y:pandas"],
#       "dependencies": ["pandas"],
#       "readme" : "",
#       "license" : "",
#	"atts": ["min", "max", "mean", "median", "std"]
# }
import pandas as pd
def ATMDataStats(ATM):
	X = ATM.inputs["X"]
	ATM.report({ 'name': "stats", 'stats': { 'min': X.min(axis=0).tolist(), 'max': X.max(axis=0).tolist(), 'mean': X.mean(axis=0).tolist(), 'median': X.median(axis=0).tolist(), 'std': X.std(ddof=0, axis=0).tolist() } })
	ATM.output(ATM.inputs);



