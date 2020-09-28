
# @component {
#	"kind" : "stats",
#	"language" : "py",
#	"description" : "Plot images from list",
#	"permissions": "public",
#	"properties": [
#		{ "name": "Colormap", "field": "colormap", "kind":"menu", "choices": ["gray", "color"], "default": "gray" },
#		{ "name": "Number to plot", "field": "count", "kind":"integer", "min": 1, "max": 100, "default": 9 }
#	],
#	"inputs": ["X:img[]", "y:string[]"],
#	"outputs": ["X:img[]", "y:string[]"],
#	"dependencies": ["matplotlib"],
#	"readme" : "",
#	"license" : ""
# }
import matplotlib.pyplot as plt
def ATMPlotImages(ATM):
	X = ATM.inputs["X"]
	for i in range(ATM.props["count"]):
		pyplot.subplot(330 + 1 + i)
		pyplot.imshow(X[i], cmap=pyplot.get_cmap(ATM.props["colormap"]))
	ATM.reportPlot()
	ATM.output(ATM.inputs);

