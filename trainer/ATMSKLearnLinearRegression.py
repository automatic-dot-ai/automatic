
# 
# @component {
#	"kind" : "trainer",
#	"language" : "py",
#	"description" : "Trains a linear regression model",
#	"permissions": "public",
#	"inputs": ["X:pandas", "y:pandas"],
#	"outputs": ["X:pandas", "y:pandas"],
#	"dependencies": ["pandas", "sklearn"],
#	"readme" : "",
#	"license" : ""
# }
from sklearn.linear_model import LinearRegression
import pickle
def ATMSKLearnLinearRegression(ATM):
	X = ATM.inputs["X"]
	y = ATM.inputs["y"]
	model = LinearRegression()
	model.fit(X, y)
	ATM.save("model", pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL))
	ATM.output({ 'X': X, 'y': y })


