

import json
import base64
from io import BytesIO
import time
import boto3
import traceback
import sys
import os
import codecs

now = lambda: int(round(time.time() * 1000))

print("ATMService.py", GLOBAL_SERVICE_URI);


def handler(event, context):
#	print('Event: {}'.format(event))
	command = event['command']
	service = event.get('service')
	pipeline = event.get('pipeline')
	payload = event 
	print("ATMService.py handler command: ", command, " for service ", service)

	os = ATMService(service)
	result = os.table()[command](pipeline, payload)
	print("returning from service after executing command:", command);
	return result or { 'message': "OK" }



class ATMService:
	def __init__(self, service):
		print("ATMService.py init", time.strftime("%Y-%m-%d %H:%M"))
		self.service = service

	def train(self, pipeline, payload):
		print("ATMService.py train")
		self.init(self.service, pipeline);
		atm = ATM(self, self.service, pipeline, -1, payload)
		atm.output(None)
		self.result['finished'] = True
		atm.report(self.result)
		#atm.report({ "name": "progress", 'purpose': "train", 'finished': True })
		print("Returning result from training", flush=True);
		return self.result

	def test(self, pipeline, payload):
		print("ATMService.py test")
		self.init(self.service, pipeline);
		atm = ATM(self, self.service, pipeline, -1, payload)
		atm.output(None)
		self.result['finished'] = True
		atm.report(self.result)
		#atm.report({ "name": "progress", 'purpose': "test", 'finished': True })
		print("Returning result from testing", flush=True);
		return self.result

	def predict(self, pipeline, payload):
		print("ATMService.py predict")
		self.init(self.service, pipeline);
		atm = ATM(self, self.service, pipeline, -1, payload)
		atm.output(payload)
		self.result['finished'] = True
		atm.report(self.result)
		#atm.report({ "name": "progress", 'purpose': "predict", 'finished': True })
		print("Returning result from predicting", flush=True);
		return self.result

	# only called if mocking
	def ping(self, pipeline, payload):
		print("service uri ponging",  GLOBAL_SERVICE_URI);
		return { 'errno': 0, 'message': GLOBAL_SERVICE_URI } ### "pong" }

	# only called if mocking
	def kill(self, pipeline, payload):
		sys.exit() #doesn't always work
		os._exit(0);
		exit()
		quit()
		return { 'errno': 0, 'message': "killed" }

	def table(self):
		return { 'train': self.train, 'test': self.test, 'predict': self.predict, 'ping': self.ping, 'kill': self.kill }

	def init(self, service, pipeline):
		self.result = { 'errno': 0, 'message': "Service " + service["uri"] + " " + pipeline["purpose"] + " pipeline completed", 'purpose': pipeline["purpose"] }

	def setReturnable(self, payload, errno=0):
		payload["errno"] = errno
		self.result = payload



class ATM:
	def __init__(self, manager, service, pipeline, index, inputs):
		self.manager = manager
		self.service = service
		self.pipeline = pipeline
		self.index = index
		self.inputs = inputs or {}
		self.stopped = False
#		print('ATM constructor.inputs: {}'.format(self.inputs))
		self.props = {}
		if index >= 0:
			self.props = pipeline["nodes"][index]

	def output(self, payload):
		print('ATM output for node #', self.index)
		if self.index + 1 < len(self.pipeline["nodes"]):
			self.index = self.index + 1
			node = self.pipeline["nodes"][self.index]
			if self.isStopped() is True:
				print("STOPPED - Stopped by request at: ", node["type"])
				self.report({ 'name': "log", 'level': "error", 'message': "Service " + self.service["uri"] + " stopped by request" })
				self.manager.setReturnable({ 'name': "log", 'level': "error", 'message': "Service " + self.service["uri"] + " " + self.pipeline["purpose"] + " pipeline stopped by request", 'purpose': self.pipeline["purpose"] }, 1)
				return
				
			atm = ATM(self.manager, self.service, self.pipeline, self.index, payload)
			print('node.type=',node["type"])
			klassname = "_".join(node["type"].split('/')).replace(":", "_")
			print('klassname=',klassname)
			f = globals().get(klassname)
			if f == None:
				print("ERROR - Implementation not found: ", node["type"])
				self.report({ 'name': "log", 'level': "error", 'message': "Implementation not found: " + node["type"], 'uri': node["uri"] })
				self.manager.setReturnable({ 'name': "log", 'level': "error", 'message': "Implementation not found: " + node["type"], 'uri': node["uri"] }, 1)
				return
			print('klassname=',klassname,'fname=',f)
			defaults = component_type_to_property_defaults.get(node["type"])
			# shallow copy
			props = dict(node)
			if defaults:
				for name in defaults:
					props[name] = node.get(name) or defaults.get(name) or None
					print("assigned default value to prop", name, props[name])
			atm.props = props
			try:
				print("executing component:", klassname);
				f(atm)
				print("returning from function after executing component:", klassname);
			except:
				self.report({ 'name': "log", 'level': "error", 'message': "Component " + klassname + " failed to execute properly\n" + traceback.format_exc(), 'at': klassname, 'uri': node["uri"] })
				self.manager.setReturnable({ 'name': "log", 'level': "error", 'message': "Component " + klassname + " failed to execute properly\n" + traceback.format_exc(), 'at': klassname, 'uri': node["uri"] }, 1)
				traceback.print_exc(file=sys.stderr)
				print("returning error from function after executing component:", klassname, traceback.format_exc());
		else:
			self.manager.setReturnable({ 'errno': 0, 'message': "Service " + self.service["uri"] + " " + self.pipeline["purpose"] + " pipeline completed", 'results': payload.get("results"), 'purpose': self.pipeline["purpose"] })


	def reportPlot(self, plt):
		buf = BytesIO()
		# requires PIL
		plt.savefig(buf, format="png")
		pixels = base64.b64encode(buf.getbuffer()).decode("ascii")
		self.report({ 'name': "image", 'image': pixels, 'format': "png" })

	def error(self, msg):
		print("E[1] ERROR: " + msg);
		self.report({ 'name': "log", 'level': "error", 'message': msg })
		self.manager.setReturnable({ 'name': "log", 'level': "error", 'message': msg }, 1)

	def report(self, payload):
		if isinstance(payload, str):
			payload = { 'name': "log", 'message': payload }
		payload = { 'command': "report", 'service_uri': self.service["uri"], 'service_token': self.service["service_token"], 'payload': payload, 'at': self.pipeline["nodes"][self.index]["type"], 'when': now(), 'uri': self.pipeline["nodes"][self.index]["uri"] }
		self._api(payload)
	
	def set(self, key, value):
		payload = { 'command': "set", 'payload': { key: key, value: value }, 'service_uri': self.service["uri"], 'service_token': self.service["service_token"] }
		self._api(payload)

	def get(self, key):
		payload = { 'command': "get", 'payload': { key: key }, 'service_uri': self.service["uri"], 'service_token': self.service["service_token"] }
		value = self._api(payload, True)
		return value

	def isStopped(self):
		if not self.stopped:
			payload = { 'command': "stopped", 'payload': { }, 'service_uri': self.service["uri"], 'service_token': self.service["service_token"] }
			self.stopped = self._api(payload, True)
			print("stopped?", self.stopped)
		return self.stopped


	def getTmpFilename(self, root):
		return ("" if MOCKING else "/tmp/") + self.service["uri"].replace("/", "-") + "-" + root.replace("/", "-").replace(":", "-").replace(".", "-") # + str(now()))

	def save(self, key, value):
		print("save ", key, " value type", type(value))
		cachekey = "saved" + key
		if key[:3] == "s3:":
			if key[:5] == "s3://":
				key = key[5:]
				bucket = key.split("/")[0]
				key = key.split("/")[1]
			else:
				key = key[3:]
				bucket = AUTOMATIC_S3_DATA_BUCKET
			filename = self.getTmpFilename(key)
		else:
			filename = self.getTmpFilename(key)
			key = self.service["uri"] + "/" + key
			bucket = AUTOMATIC_S3_DATA_BUCKET

		print("saving data to local file:", filename)
		try:
			# Try to save a cached local copy
			with open(filename, "wb" if isinstance(value, bytes) else "w") as file:
				file.write(value)
				self.service[cachekey] = filename
		except:
			pass
		bucket = AUTOMATIC_S3_DATA_BUCKET
		try:
			client = boto3.client('s3')
			client.put_object(Body=value, Bucket=bucket, Key=key)
		except:
			self.error("Unable to save data to S3: " + bucket + "/" + key + " [ " + traceback.format_exc() + " ]")

	def load(self, key, is_binary=False):
		cachekey = "saved" + key
		if self.service.get(cachekey) is not None:
			print("ATMService load - returning cached data", self.service.get(cachekey))
			return self.service.get(cachekey)

		## TODO support https:// keys
		if key[:3] == "s3:":
			if key[:5] == "s3://":
				key = key[5:]
				bucket = key.split("/")[0]
				key = key[len(bucket) + 1:]
				filename = self.getTmpFilename(bucket + "_" + key)
			else:
				key = key[3:]
				bucket = AUTOMATIC_S3_DATA_BUCKET
				filename = self.getTmpFilename(bucket + "_" + key)
		else:
			bucket = AUTOMATIC_S3_DATA_BUCKET
			filename = self.getTmpFilename(bucket + "_" + key)
			key = self.service["uri"] + "/" + key
		s3 = boto3.resource('s3')
		print( "downloading from s3: " + bucket + "/" + key + " -> " + filename )
		try:
			s3.meta.client.download_file(bucket, key, filename)
			self.service[cachekey] = filename
		except:
			self.error("Unable to load from S3: " + bucket + "/" + key + " [ " + traceback.format_exc() + " ]")
		return filename


	def _api(self, payload, synchronous=False):
		if 'MockInvoke' in globals():
			return MockInvoke.do_POST(self.service, payload)

		synchronous = True ## TESTT debug
		print("service: " + self.service["uri"] + " calling API: " + json.dumps(payload))
		lambda_client = boto3.client('lambda')
		invoke_response = lambda_client.invoke(
					FunctionName = "automatic-api",
					InvocationType = 'RequestResponse' if synchronous else 'Event',
					LogType = 'None',
					Payload = json.dumps( { 'command': "serviceFeedback", 'payload': payload } )
				)
		'''
		   data = {
		    FunctionError: "", 
		    LogResult: "", 
		    Payload: <Binary String>, 
		    StatusCode: 123
		   }
		'''
		print("api request", payload, "returned: ", invoke_response)
		payload = invoke_response["Payload"]
		return payload


