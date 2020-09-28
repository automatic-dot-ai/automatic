
# @component {
#	"kind" : "datasource",
#	"language" : "py",
#	"description" : "",
#	"permissions": "public",
#	"properties": [
#		{ "name": "URL" , "field": "url", "kind": "url", "minlen": 8, "maxlen": 256, "required": true },
#		{ "name": "Target" , "field": "target", "kind": "string", "minlen": 1, "maxlen": 62, "required": true, "hint": "The column to treat as the target feature. '0' indicates leftmost column. This is usually specified in the associated dataset. If not, the Notebook's preview dump of the data may provide a clue about the name of the feature." }
#	],
#	"outputs": ["X:pandas", "y:pandas"],
#	"dependencies": ["pandas"],
#	"readme" : "",
#	"license" : ""
# }
import io
import boto3
import traceback
import pandas as pd
from io import BytesIO
def ATMCSVDataSource(ATM):
	# csv and zipped csv files
	url = ATM.props["url"]
	print("ATMCSVDataSource downloading from url=", url);
	try:
		src = url
		compression = "infer"
		if url[0:5] == "s3://":
			url = url[5:]
			s3_obj = boto3.resource('s3').Object(bucket_name=url[:url.index("/")], key=url[url.index("/")+1:])
			src = BytesIO(s3_obj.get()["Body"].read())
			compression = None
			if url.rindex(".") != -1:
				compression = url[url.rindex(".") + 1:]
			if compression == "gz":
				compression = "gzip"
		data = pd.read_csv(src, compression=compression)
	except:
		return ATM.error("Unable to read csv file at url '" + url + "' at ATMCSVDataSource [ " + traceback.format_exc() + " ]")

	target = ATM.props["target"]
	if target == "0":
		X = data.iloc[:, 1:]
		y = data.iloc[:, 0]
	else:
		y = data[target]
		X = data.drop(target, axis = 1)
	ATM.report({ 'name': "stats", 'stats': { 'features': list(X.columns) if target != "0" else [], 'target': target, 'rows': X.shape[0] } })
	ATM.report("Loaded dataset with {0} data points with {1} variables each".format(*X.shape))
	ATM.output({ 'X': X, 'y': y })

