
# @component {
#	"kind" : "modelsource",
#	"language" : "py",
#	"description" : "This is for models whose weights data are hosted on AWS S3.",
#	"permissions": "public",
#	"properties": [
#		{"name":"Name","field":"name","kind":"string","minlen":8,"maxlen":64,"placeholder":"","required":true},
#		{"name":"Format","field":"format","kind":"radio","required":true,"default":"h5","choices":["h5","onnx","pkl","pt"]},
#		{ "name": "Framework", "field": "framework", "kind": "menu", "choices": ["sklearn", "tensorflow", "pytorch"], "hint": "Only one framework can be used by a service", "required": true, "protected": true, "default": "pytorch" },
#		{"name":"Link","field":"link","kind":"url","minlen":12,"maxlen":256,"placeholder":"Link to model data","required":false,"disabled":true, "hint": "Link to uploaded model (after uploading)"},
#		{"name":"Size","field":"size","kind":"string","minlen":4,"maxlen":256,"placeholder":"Size of model","required":false}
#	],
#	"dependencies": [],
#	"readme" : "",
#	"license" : ""
# }
def ATMS3ModelSource(ATM):
	model_uri = ATM.props("link")
	model = ATM.load(model_uri, True)
	# move to standard model location for services: S3:<service_uri>/model
	ATM.save("model", model)
