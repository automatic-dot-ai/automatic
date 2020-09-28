# 
# @component {
#	"kind" : "predictor",
#	"language" : "py",
#	"description" : "Performs single-valued prediction using previous trained model",
#	"permissions": "public",
#	"inputs": ["data:pandas"],
#	"outputs": ["output:dict"],
#	"dependencies": ["pandas"],
#	"readme" : "",
#	"license" : ""
# }
import pickle
def ATMSKLearnPredictor(ATM):
#old	model = pickle.loads(ATM.load("model"))
	with open(ATM.load("model"), "rb") as f:
		model = pickle.load(f)
	results = model.predict(ATM.inputs["data"])
	results = results[0][0];
	ATM.report({ 'name': "predictions", 'predictions': results })
	ATM.output({ 'results': results });
		

