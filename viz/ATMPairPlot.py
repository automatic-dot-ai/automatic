# 
# @component {
#	"kind" : "viz",
#	"language" : "py",
#	"description" : "Plots for each feature vs. other features",
#	"permissions": "public",
#	"inputs": ["X:pandas", "y:pandas"],
#	"outputs": ["X:pandas", "y:pandas"],
#	"dependencies": ["pandas", "matplotlib", "seaborn"],
#	"readme" : "",
#	"license" : ""
# }
import matplotlib.pyplot as plt
import seaborn as sns

def ATMPairPlot(ATM):
	X = ATM.inputs["X"]
	sns.pairplot(X, size=2.5)
	plt.tight_layout()
	ATM.reportPlot(plt)
	ATM.output(ATM.inputs);


