# @component {
#	"kind" : "stats",
#	"language" : "py",
#	"description" : "First few rows of dataset",
#	"permissions": "public",
#	"inputs": ["X:pandas", "y:pandas"],
#	"outputs": ["X:pandas", "y:pandas"],
#	"dependencies": ["pandas"],
#	"readme" : "",
#	"license" : "",
#	"atts": ["head"]
# }
def ATMDataHead(ATM):
	X = ATM.inputs["X"]
	ATM.report({ 'name': "stats", 'stats': { 'head': X.head().to_json() } })
	ATM.output(ATM.inputs);
		

