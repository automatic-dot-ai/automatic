#
# @component {
#       "kind" : "tester",
#       "language" : "py",
#	"description": "Tests a model",
#       "permissions": "public",
#       "properties": [
#       ],
#       "inputs": ["X:pandas", "y:pandas"],
#       "outputs": ["X:pandas", "y:pandas"],
#       "dependencies": ["pandas", "tensorflow", "sklearn"],
#       "readme" : "",
#       "license" : "",
#	"links": ["https://dataoutpost.wordpress.com/2018/05/28/housing-price-prediction-with-multi-layer-perceptron/"]
# }
import pandas as pd
import matplotlib.pyplot as plt


def ATMKerasModelTester(ATM):
	X = ATM.inputs["X"]
	y = ATM.inputs["y"]
	model = load_model(ATM.load("model"))
	acc = model.evaluate(X_test, Y_test)[1]
	ATM.report({ 'name': "stats", 'stats': { 'accuracy': acc } })


