
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
def ATMTargetDataStats(ATM):
	y = ATM.inputs["y"]
	ATM.report({ 'name': "stats", 'stats': { 'min': y.min(), 'max': y.max(), 'mean': y.mean(), 'median': y.median(), 'std': y.std(ddof=0) } })
	ATM.output(ATM.inputs);



