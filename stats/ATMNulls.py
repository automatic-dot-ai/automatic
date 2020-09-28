# @component {
#	"kind" : "stats",
#	"language" : "py",
#	"description" : "Number of null values",
#	"permissions": "public",
#	"inputs": ["X:pandas", "y:pandas"],
#	"outputs": ["X:pandas", "y:pandas"],
#	"dependencies": ["pandas"],
#	"readme" : "",
#	"license" : "",
#	"atts": ["nulls"]
# }
def ATMNulls(ATM):
	X = ATM.inputs["X"]
	ATM.report({ 'name': "stats", 'stats': { 'nulls': X.isnull().sum() } })
	ATM.output(ATM.inputs);


