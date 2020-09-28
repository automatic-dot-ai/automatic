
# @component {
#	"kind" : "viz",
#	"language" : "py",
#	"description" : "Plot of loss vs. val_loss",
#	"permissions": "public",
#	"properties": [
#		{ "name": "parm1" , "field": "parm1", "kind": "string", "minlen": 1, "maxlen": 62, "required": true, "default": "acc" },
#		{ "name": "parm2" , "field": "parm2", "kind": "string", "minlen": 1, "maxlen": 62, "required": true, "default": "val_acc" }
#	],
#	"inputs": ["X:pandas", "y:pandas", "history:dict" ],
#	"outputs": ["X:pandas", "y:pandas", "history:dict" ],
#	"dependencies": ["matplotlib"],
#	"readme" : "",
#	"license" : ""
# }
import matplotlib.pyplot as plt
def ATMAccValAccPlot(ATM):
	history = ATM.inputs["history"]
	plt.plot(history[ATM.props['parm1']])
	plt.plot(history[ATM.props['parm2']])
	plt.title('Model accuracy')
	plt.ylabel('Accuracy')
	plt.xlabel('Epoch')
	plt.legend(['Train', 'Val'], loc='lower right')
	plt.ylim(top=1.2, bottom=0)
	ATM.reportPlot(plt)
	ATM.output(ATM.inputs);

