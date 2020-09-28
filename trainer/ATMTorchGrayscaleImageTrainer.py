
# @component {
#	"kind" : "trainer",
#	"language" : "py",
#	"description" : "Train model to recognize categories of grayscale images (MNIST)",
#	"permissions": "public",
#	"properties": [
#		{ "name": "Epochs" , "field": "epochs", "kind": "integer", "min": 1, "max": 1000, "required": true, "default": 4, "hint": "Pytorch recommends 14 epochs, which is about 20 minutes training time on a CPU" },
#		{ "name": "Batch size" , "field": "batch_size", "kind": "integer", "min": 2, "max": 1000, "required": true, "default": 64 },
#		{ "name": "Learning rate" , "field": "learning_rate", "kind": "number", "min": 0.0001, "max": 1, "required": true, "default": 1.0 },
#		{ "name": "Learning rate gamma" , "field": "learning_rate_gamma", "kind": "number", "min": 0.0001, "max": 1, "required": true, "default": 0.7 },
#		{ "name": "Random seed" , "field": "seed", "kind": "number", "required": true, "default": 1 }
#	],
#	"inputs": ["X:pandas", "y:pandas"],
#	"outputs": ["X:pandas", "y:pandas"],
#	"dependencies": ["pytorch", "numpy", "sklearn"],
#	"readme" : "",
#	"license" : "",
#	"links" : [ "https://github.com/pytorch/examples/blob/master/mnist/main.py" ]
# }
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import numpy as np
from sklearn.model_selection import train_test_split


def ATMTorchGrayscaleImageTrainer(ATM):
	img_rows = ATM.props.get("height") or 28
	img_cols = ATM.props.get("width") or 28
	X_train = ATM.inputs["X"]
	y_train = ATM.inputs["y"]

	num_columns = X_train.columns
	num_rows = X_train.shape[0]

	X_train, X_validate, y_train, y_validate = train_test_split(X_train, y_train, test_size = 0.2, random_state = 1212)

	X_train = np.array(X_train).reshape(X_train.shape[0], img_rows*img_cols)
	X_validate = np.array(X_validate).reshape(X_validate.shape[0], img_rows*img_cols)

	X_train = X_train.astype('float32') / 255.0
	X_validate = X_validate.astype('float32') / 255.0

	X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
	X_train  = torch.from_numpy(X_train) ## .float()
	y_train = torch.from_numpy(np.array(y_train))

	X_validate = X_validate.reshape(X_validate.shape[0], 1, img_rows, img_cols)
	X_validate  = torch.from_numpy(X_validate) ## .float()
	y_validate = torch.from_numpy(np.array(y_validate))


	train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
	validate_dataset = torch.utils.data.TensorDataset(X_validate, y_validate)

	torch.manual_seed(ATM.props.get("seed"))

	use_cuda = ATM.props.get("GPU") and torch.cuda.is_available()
	device = torch.device("cuda" if use_cuda else "cpu")

	kwargs = {'batch_size': ATM.props.get("batch_size")}
	if use_cuda:
		kwargs.update({'num_workers': 1, 'pin_memory': True, 'shuffle': True},)

	train_loader = torch.utils.data.DataLoader(train_dataset, **kwargs)
	test_loader = torch.utils.data.DataLoader(validate_dataset, **kwargs)

	model = Net().to(device)

	optimizer = optim.Adadelta(model.parameters(), lr=ATM.props.get("learning_rate"))
	scheduler = StepLR(optimizer, step_size=1, gamma=ATM.props.get("learning_rate_gamma"))

	for epoch in range(1, ATM.props["epochs"] + 1):
		train(ATM, model, device, train_loader, optimizer, epoch)
		test(ATM, model, device, test_loader)
		scheduler.step()
		if ATM.props.get("dry_run"):
			break
		if ATM.isStopped():
			ATM.report("Stopped training by request at epoch: " + str(epoch))
			break

	filename = ATM.getTmpFilename("model.pt")
	print("saving model to filename:", filename)
	torch.save(model.state_dict(), filename)
	model_pt = open(filename, 'rb').read()
	ATM.save("model", model_pt)
	ATM.report({ "name": "progress", 'purpose': "train", 'finished': True })
	ATM.output(ATM.inputs);


class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.conv1 = nn.Conv2d(1, 32, 3, 1)
		self.conv2 = nn.Conv2d(32, 64, 3, 1)
		self.dropout1 = nn.Dropout2d(0.25)
		self.dropout2 = nn.Dropout2d(0.5)
		self.fc1 = nn.Linear(9216, 128)
		self.fc2 = nn.Linear(128, 10)

	def forward(self, x):
		x = self.conv1(x)
		x = F.relu(x)
		x = self.conv2(x)
		x = F.relu(x)
		x = F.max_pool2d(x, 2)
		x = self.dropout1(x)
		x = torch.flatten(x, 1)
		x = self.fc1(x)
		x = F.relu(x)
		x = self.dropout2(x)
		x = self.fc2(x)
		output = F.log_softmax(x, dim=1)
		return output


def train(ATM, model, device, train_loader, optimizer, epoch):
	model.train()
	for batch_idx, (data, target) in enumerate(train_loader):
		data, target = data.to(device), target.to(device)
		optimizer.zero_grad()
		output = model(data)
		loss = F.nll_loss(output, target)
		loss.backward()
		optimizer.step()
	ATM.report({ "name": "progress", 'purpose': "train", 'progress': epoch / ATM.props["epochs"], 'loss': loss.item(), 'finished': False }),

def test(ATM, model, device, test_loader):
	model.eval()
	test_loss = 0
	correct = 0
	with torch.no_grad():
		for data, target in test_loader:
			data, target = data.to(device), target.to(device)
			output = model(data)
			test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
			pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
			correct += pred.eq(target.view_as(pred)).sum().item()

	test_loss /= len(test_loader.dataset)

	ATM.report({ 'name': "log", 'message': 'Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
		test_loss, correct, len(test_loader.dataset),
		100. * correct / len(test_loader.dataset)) })
	ATM.report({ 'name': "stats", 'stats': { 'accuracy': correct / len(test_loader.dataset) } })


