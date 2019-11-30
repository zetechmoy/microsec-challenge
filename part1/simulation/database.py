import time

class DatabaseSimulator(object):
	"""
	Simulate a SQL Database. I could have used a real database but
	for time and compatibility when user gonna run it reasons,
	i chose to simulate.
	"""

	#used as a table having rows like (id, seconds, publisher, temperature)
	content = dict()

	def __init__(self):
		"""Init DatabaseSimulator with some default values"""
		super(DatabaseSimulator, self).__init__()

		#init table
		self.content["temperatures"] = list()

	def insert(self, publisher, temperature):
		"""
		Add a entry to the table, SQL would be like
		INSERT INTO temperatures(id, secs, publisher, temperature)
		VALUES (NULL, time.time(), publisher, temperature)
		"""
		id = 1
		while id in [item["id"] for item in self.content["temperatures"]]:
			id += 1

		data = {"id": id,"secs":time.time(), "publisher":publisher, "temperature":temperature}
		self.content["temperatures"].append(data)
		return data

	def select(self, ):
		"""
		Get the table, SQL would be like
		SELECT * FROM temperatures;
		"""
		return self.content
