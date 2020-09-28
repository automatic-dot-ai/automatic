
# 
# @component {
#	"kind" : "viz",
#	"language" : "py",
#	"description" : "Charts actual vs. predicted values",
#	"permissions": "public",
#	"inputs": ["X:pandas", "y:pandas", "y_predicted:pandas"],
#	"outputs": ["X:pandas", "y:pandas"],
#	"dependencies": ["pandas", "sklearn"],
#	"readme" : "",
#	"license" : ""
# }
def ATMLinearRegressionResults(ATM):
	y = ATM.inputs["y"]
	if y_predicted not in ATM.inputs:
		return ATM.report({'name': "log", "level":"error", "msg":"y_predicted array not found", "at":"ATMLinearRegressionResults"});
	y_pred = ATM.inputs["y_predicted"]
	plt.scatter(y, y_pred)
	plt.xlabel("Actual")
	plt.ylabel("Predicted")
	plt.xticks(range(0, int(max(y)),2))
	plt.yticks(range(0, int(max(y)),2))
	plt.title("Actual vs Predicted")
	ATM.reportPlot(plt)
	ATM.output(ATM.inputs);


