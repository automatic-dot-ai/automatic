# @component {
#	"kind" : "stats",
#	"language" : "py",
#	"description" : "Description of dataset values",
#	"permissions": "public",
#	"inputs": ["X:pandas", "y:pandas"],
#	"outputs": ["X:pandas", "y:pandas"],
#	"dependencies": ["pandas"],
#	"readme" : "",
#	"license" : "",
#	"atts": ["describe"]
# }
def ATMDescribe(ATM):
	X = ATM.inputs["X"]
	ATM.report({ 'name': "stats", 'stats': { 'describe': X.describe().to_json() } })
	ATM.output(ATM.inputs);
		

