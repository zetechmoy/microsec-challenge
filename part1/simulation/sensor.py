
import threading, random, time, requests

class TemperatureSensorSimulator(threading.Thread):
	"""
	Simulate a temperature sensor measuring every 10 secs
	"""

	def __init__(self, name):
		super(TemperatureSensorSimulator, self).__init__()
		self.name = name
		self.start_time = time.time()

	def measure(self):
		data = {"name":self.name, "temperature": random.uniform(0,30)}
		requests.get(url = "http://localhost:5000/insert", params = data)

	def run(self):
		while True:
			if time.time()-self.start_time >= 5:
				self.start_time = time.time()
				self.measure()
