
# @component {
#	"kind" : "splitter",
#	"language" : "py",
#       "description" : "Split supervised datasets into training and testing datasets. Split is accross train and test pipelines: this component needs to appear in both in order to work. The test data is split off from the dataset and saved during training and then used as the X, y during testing",
#	"permissions": "public",
#	"properties": [
#		{ "name": "Test dataset" , "field": "test", "kind": "boolean", "required": false, "hint": "Set to true if want to use test dataset" },
#		{ "name": "Test proportion" , "field": "test_size", "kind": "number", "min": 0.01, "max": 0.99, "required": true, "default": 0.2 },
#		{ "name": "Random state" , "field": "random_state", "kind": "number", "min": 1, "max": 99, "required": true, "default": 10 }
#	],
#	"inputs": ["X:pandas", "y:pandas"],
#	"outputs": ["X:pandas", "y:pandas", "X_test:pandas", "y_test:pandas"],
#	"dependencies": ["sklearn"],
#	"readme" : "",
#	"license" : ""
# }
from sklearn.cross_validation import train_test_split
def ATMCSVTrainTestSplit(ATM):
	X = ATM.inputs["X"]
	y = ATM.inputs["y"]
	if ATM.props.get("test"):
		dataset = ATM.get("datatest")
		dataset.X = dataset.X_test
		dataset.y = dataset.y_test
		ATM.output(dataset)
	else:
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ATM.props["test_size"], random_state=ATM.props["random_state"])
		dataset = { 'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test }
		dataset.X = dataset.X_train
		dataset.y = dataset.y_train
		ATM.set("datatest", { 'X_test': X_test, 'y_train': y_train })
		ATM.output({ 'X': X, 'y': y, 'X_test': X_test, 'y_test': y_test })
