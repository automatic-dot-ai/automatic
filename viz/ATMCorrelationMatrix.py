# 
# @component {
#	"kind" : "viz",
#	"language" : "py",
#	"description" : "Plots for each feature correlation matrix",
#	"permissions": "public",
#	"inputs": ["X:pandas", "y:pandas"],
#	"outputs": ["X:pandas", "y:pandas"],
#	"dependencies": ["pandas", "matplotlib", "seaborn"],
#	"readme" : "",
#	"license" : ""
# }
import matplotlib.pyplot as plt
import seaborn as sns

def ATMCorrelationMatrix(ATM):
	X = ATM.inputs["X"]
	cm = np.corrcoef(X.values.T)
	sns.set(font_scale=1.5)
	hm = sns.heatmap(cm,
		cbar=True,
		annot=True,
		square=True,
		fmt='.2f',
		annot_kws={'size': 15},
		yticklabels=cols,
		xticklabels=cols)
	ATM.reportPlot(plt)
	ATM.output(ATM.inputs);



