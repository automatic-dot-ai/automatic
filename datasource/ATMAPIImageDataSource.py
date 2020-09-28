
# @component {
#	"kind" : "datasource",
#	"language" : "py",
#	"description" : "Data is supplied by the API for training, testing or prediction. For example, using a user-interface or external program to supply the dataset for a pipeline",
#	"permissions": "public",
#	"outputs": ["data:img"],
#	"dependencies": [],
#	"readme" : "",
#	"license" : ""
# }
def ATMAPIImageDataSource(ATM):
	ATM.output(ATM.inputs)
