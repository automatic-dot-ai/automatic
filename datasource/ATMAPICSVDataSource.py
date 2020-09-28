
# @component {
#	"kind" : "datasource",
#	"language" : "py",
#	"description" : "Data is supplied by the API for training, testing or prediction. For example, using a user-interface or external program to supply the dataset for a pipeline",
#	"permissions": "public",
#	"outputs": ["data:pandas"],
#	"dependencies": ["pandas"],
#	"readme" : "",
#	"license" : ""
# }
import pandas as pd
from io import StringIO
def ATMAPICSVDataSource(ATM):
	data = pd.read_csv(StringIO(ATM.inputs["data"]))
	ATM.output({ 'data': data })
