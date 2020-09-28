
# @component {
#	"kind" : "predictor",
#	"language" : "py",
#	"description" : "Performs stylization of one image from a previously trained model representing another 'artistic' image",
#	"permissions": "public",
#	"properties": [
#		{ "name": "Pixel width" , "field": "width", "kind": "integer", "min": 28, "max": 1000, "required": true, "default": 500 },
#		{ "name": "Pixel height" , "field": "height", "kind": "integer", "min": 28, "max": 1000, "required": true, "default": 500 },
#		{ "name": "Model" , "field": "model_uri", "kind": "string", "required": false, "hint": "URI of previously trained model of 'artisitic' image representing the style" },
#		{ "name": "cuda" , "field": "cuda", "kind": "boolean", "required": false, "default": false }
#	],
#	"inputs": ["data:img"],
#	"outputs": ["results:img"],
#	"dependencies": ["pytorch", "pillow", "numpy"],
#	"readme" : "",
#	"license" : "",
#	"links": ["https://github.com/pytorch/examples/tree/master/fast_neural_style"]
# }

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets
from torchvision import transforms
import numpy as np

import re
import base64
import PIL
from PIL import Image
from io import BytesIO


global_style_model = {}

def ATMTorchImageStyleTransfer(ATM):
	content_image = Image.fromarray(np.array(ATM.inputs["data"], dtype=np.uint8), "RGB")
	use_cuda = ATM.props.get("cuda") and torch.cuda.is_available()
	device = torch.device("cuda" if use_cuda else "cpu")

	content_transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Lambda(lambda x: x.mul(255))
	])
	content_image = content_transform(content_image)
	content_image = content_image.unsqueeze(0).to(device)

	with torch.no_grad():
		global global_style_model
		model_uri = ATM.inputs.get("props").get("model_uri")
		if global_style_model.get(model_uri) == None:
			m = TransformerNet()
			state_dict = torch.load(ATM.load(model_uri, True))
			if state_dict is None:
				return ATM.report({ 'name': "log", 'level': "error", 'message': "Styling model at " + model_uri + " not found", 'at': "ATMTorchImageStyleTransfer" })
			for k in list(state_dict.keys()):
				if re.search(r'in\d+\.running_(mean|var)$', k):
					del state_dict[k]
			m.load_state_dict(state_dict)
			global_style_model[model_uri] = m
		m = global_style_model.get(model_uri)
		m.to(device)
		output = m(content_image).cpu()

	img = output[0].clone().clamp(0, 255).numpy()
	img = img.transpose(1, 2, 0).astype("uint8")
	img = Image.fromarray(img)
	buf = BytesIO()
	img.save(buf, "jpeg")
	result = base64.b64encode(buf.getbuffer()).decode("ascii")
	filename = ATM.inputs.get("props").get("output_image_uri") + ".jpg"
	uri = ATM.service["uri"] + "/" + filename
	print("uri for img", uri, flush=True)
	# handle async calls - notify about results
	ATM.report({ 'name': "image", 'image': result, 'filename': filename, 'format': "jpeg" })
	ATM.report({ 'name': "predictions", 'predictions': uri, 'finished': True })
	# handle sync calls - return results
	ATM.output({ 'results': uri })



class TransformerNet(torch.nn.Module):
    def __init__(self):
        super(TransformerNet, self).__init__()
        # Initial convolution layers
        self.conv1 = ConvLayer(3, 32, kernel_size=9, stride=1)
        self.in1 = torch.nn.InstanceNorm2d(32, affine=True)
        self.conv2 = ConvLayer(32, 64, kernel_size=3, stride=2)
        self.in2 = torch.nn.InstanceNorm2d(64, affine=True)
        self.conv3 = ConvLayer(64, 128, kernel_size=3, stride=2)
        self.in3 = torch.nn.InstanceNorm2d(128, affine=True)
        # Residual layers
        self.res1 = ResidualBlock(128)
        self.res2 = ResidualBlock(128)
        self.res3 = ResidualBlock(128)
        self.res4 = ResidualBlock(128)
        self.res5 = ResidualBlock(128)
        # Upsampling Layers
        self.deconv1 = UpsampleConvLayer(128, 64, kernel_size=3, stride=1, upsample=2)
        self.in4 = torch.nn.InstanceNorm2d(64, affine=True)
        self.deconv2 = UpsampleConvLayer(64, 32, kernel_size=3, stride=1, upsample=2)
        self.in5 = torch.nn.InstanceNorm2d(32, affine=True)
        self.deconv3 = ConvLayer(32, 3, kernel_size=9, stride=1)
        # Non-linearities
        self.relu = torch.nn.ReLU()

    def forward(self, X):
        y = self.relu(self.in1(self.conv1(X)))
        y = self.relu(self.in2(self.conv2(y)))
        y = self.relu(self.in3(self.conv3(y)))
        y = self.res1(y)
        y = self.res2(y)
        y = self.res3(y)
        y = self.res4(y)
        y = self.res5(y)
        y = self.relu(self.in4(self.deconv1(y)))
        y = self.relu(self.in5(self.deconv2(y)))
        y = self.deconv3(y)
        return y


class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        reflection_padding = kernel_size // 2
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out


class ResidualBlock(torch.nn.Module):
    """ResidualBlock
    introduced in: https://arxiv.org/abs/1512.03385
    recommended architecture: http://torch.ch/blog/2016/02/04/resnets.html
    """

    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in1 = torch.nn.InstanceNorm2d(channels, affine=True)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in2 = torch.nn.InstanceNorm2d(channels, affine=True)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        out = out + residual
        return out


class UpsampleConvLayer(torch.nn.Module):
    """UpsampleConvLayer
    Upsamples the input and then does a convolution. This method gives better results
    compared to ConvTranspose2d.
    ref: http://distill.pub/2016/deconv-checkerboard/
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        if upsample:
            self.upsample_layer = torch.nn.Upsample(mode='nearest', scale_factor=upsample)
        reflection_padding = kernel_size // 2
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        x_in = x
        if self.upsample:
            x_in = self.upsample_layer(x_in)
        out = self.reflection_pad(x_in)
        out = self.conv2d(out)
        return out

