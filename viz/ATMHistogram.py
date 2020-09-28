
# @component {
#	"kind" : "viz",
#	"language" : "py",
#	"description" : "Histogram of target feature's values",
#	"permissions": "public",
#	"properties": [
#		{ "name": "Feature" , "field": "feature", "kind": "string", "minlen": 1, "maxlen": 62, "required": true, "feature": true },
#		{ "name": "Bins" , "field": "bins", "kind": "integer", "min": 2, "max": 100, "required": true, "default": 10 },
#		{ "name": "X-axis Label" , "field": "xlabel", "kind": "string", "minlen": 1, "maxlen": 16, "required": true, "default": "Histogram" }
#	],
#	"inputs": ["X:pandas", "y:pandas"],
#	"outputs": ["X:pandas", "y:pandas"],
#	"dependencies": ["pandas", "matplotlib"],
#	"readme" : "",
#	"license" : ""
# }
import matplotlib.pyplot as plt
import pandas as pd
def ATMHistogram(ATM):
	X = ATM.inputs["X"]
	y = ATM.inputs["y"]
	data = pd.concat([X, y], axis=1, sort=False)
	plt.hist(data[ATM.props["feature"]], bins=ATM.props["bins"])
	plt.xlabel(ATM.props["xlabel"])
	ATM.reportPlot(plt)
	ATM.output(ATM.inputs);

