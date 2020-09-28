# 
# @component {
#	"kind" : "viz",
#	"language" : "py",
#	"description" : "Plots for each feature vs. the target feature",
#	"permissions": "public",
#	"properties": [
#		{ "name": "Target" , "field": "target", "kind": "string", "minlen": 1, "maxlen": 62, "required": true, "hint": "The label to be associated with the target feature" }
#	],
#	"inputs": ["X:pandas", "y:pandas"],
#	"outputs": ["X:pandas", "y:pandas"],
#	"dependencies": ["pandas", "matplotlib"],
#	"readme" : "",
#	"license" : "",
#	"links": ["https://matplotlib.org/3.1.1/faq/howto_faq.html#howto-webapp"]
# }
from matplotlib.figure import Figure
import matplotlib.style as mplstyle
import matplotlib as mpl
mplstyle.use(['dark_background', 'ggplot', 'fast'])
# minimize path length
mpl.rcParams['path.simplify'] = True
mpl.rcParams['path.simplify_threshold'] = 1.0

def ATMPlot(ATM):
	# plt.figure(figsize=(20, 5))
	X = ATM.inputs["X"]
	targets = ATM.inputs["y"]
	numcols = ATM.inputs.shape[0]
	for i, col in enumerate(X.columns):
		fig = Figure()
		fig.subplot(1, numcols, i+1)
		x = X[col]
		y = targets
		fig.plot(x, y, 'o')
		# Create regression line
		fig.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)))
		fig.title(col)
		fig.xlabel(col)
		fig.ylabel(ATM.props["target"])
		ATM.reportPlot(fig)
	ATM.output(ATM.inputs);


