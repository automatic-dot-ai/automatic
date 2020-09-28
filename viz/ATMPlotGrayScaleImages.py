# 
# @component {
#	"kind" : "viz",
#	"language" : "py",
#	"description" : "Plots grayscale images",
#	"permissions": "public",
#	"properties": [
#		{ "name": "Number to plot" , "field": "count", "kind": "integer", "min": 1, "max": 1000, "required": true, "default": 9 },
#		{ "name": "Number of columns" , "field": "numcols", "kind": "integer", "min": 1, "max": 10, "required": true, "default": 3 }
#	],
#	"inputs": ["X:img[]", "y:string[]"],
#	"outputs": ["X:img[]", "y:string[]"],
#	"dependencies": ["matplotlib"],
#	"readme" : "",
#	"license" : "",
#	"links": ["https://machinelearningmastery.com/how-to-develop-a-cnn-from-scratch-for-fashion-mnist-clothing-classification/"]
# }
import matplotlib.pyplot as plt

def ATMPlotGrayScaleImages(ATM):
	X = ATM.inputs["X"]
	print("ATMPlotGrayScaleImages X", X)
	numtoplot = ATM.props.get("count") or 9
	numcols = ATM.props.get("columns") or 3
	for i in range(numtoplot):
		plt.subplot(330 + 1 + i) 
		plt.imshow(X[i], cmap='gray')
	ATM.reportPlot(plt) 
	plt.close('all')
	ATM.output(ATM.inputs);


