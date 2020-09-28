
# @component {
#	"kind" : "viz",
#	"language" : "py",
#	"description" : "Plot loss, val_loss, accuracy and val_accuracy and scores for trained model on k-folded input data",
#	"permissions": "public",
#	"properties": [
#		{ "name": "Num KFolds" , "field": "n_folds", "kind": "integer", "min": 2, "max": 100, "required": true, "default": 10 }
#	],
#	"inputs": ["X:pandas", "y:pandas"],
#	"outputs": ["X:pandas", "y:pandas"],
#	"dependencies": ["pandas"],
#	"readme" : "",
#	"license" : ""
# }
def ATMEvaluateModel(ATM):
	X = ATM.inputs["X"]
	y = ATM.inputs["y"]
	model = ATM.get("model.h5")
	with open("model.h5", "w") as model_file:
		model_file.write(model)
	model = load_model('model.h5')
	scores, histories = self.evaluate_model(model, X, y)
	self.summarize_diagnostics(histories)
	self.summarize_performance(scores)
	ATM.output(ATM.inputs);

	def evaluate_model(self, model, dataX, dataY, n_folds=5):
		scores, histories = list(), list()
		kfold = KFold(ATM.props["n_folds"], shuffle=True, random_state=1)
		for train_ix, test_ix in kfold.split(dataX):
			trainX, trainY, testX, testY = dataX[train_ix], dataY[train_ix], dataX[test_ix], dataY[test_ix]
			# history = model.fit(trainX, trainY, epochs=10, batch_size=32, validation_data=(testX, testY), verbose=0)
			_, acc = model.evaluate(testX, testY, verbose=0)
			#print('> %.3f' % (acc * 100.0))
			scores.append(acc)
			histories.append(history)
		return scores, histories

	def summarize_diagnostics(self, histories):
		for i in range(len(histories)):
			pyplot.subplot(2, 1, 1)
			pyplot.title('Cross Entropy Loss')
			pyplot.plot(histories[i].history['loss'], color='blue', label='train')
			pyplot.plot(histories[i].history['val_loss'], color='orange', label='test')
			pyplot.subplot(2, 1, 2)
			pyplot.title('Classification Accuracy')
			pyplot.plot(histories[i].history['accuracy'], color='blue', label='train')
			pyplot.plot(histories[i].history['val_accuracy'], color='orange', label='test')
		pyplot.show()
		ATM.reportPlot()

	def summarize_performance(self, scores):
		#print('Accuracy: mean=%.3f std=%.3f, n=%d' % (mean(scores)*100, std(scores)*100, len(scores)))
		pyplot.boxplot(scores)
		pyplot.show()
		ATM.reportPlot()


