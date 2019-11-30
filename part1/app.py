from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit

from simulation.database import DatabaseSimulator
from simulation.sensor import TemperatureSensorSimulator

#pip3 install flask flask-socketio eventlet
#inspired by https://github.com/rakibtg/Python-Chat-App

app = Flask(__name__)

app.config[ 'SECRET_KEY' ] = 'thisisaverysecretkeyforMicroSecChallenge'
socketio = SocketIO(app)
db = DatabaseSimulator()

@app.route( '/' )
def hello():
	"""simply render dashboard, dashboard will listen for 'update-sensors' events"""
	return render_template( './dashboard.html' )

@app.route( '/insert')
def insert():
	"""api endpoints allowing sensors to add values to database and update dashboard"""
	#get values sent from sensors
	sensor_name = request.args.get('name')
	sensor_temperature = request.args.get('temperature')

	#add to database and get all values
	value = db.insert(sensor_name, sensor_temperature)

	#emit socket io to send values to dashboard
	socketio.emit('update-sensors', value)
	return {"status": 1, "values":value}

if __name__ == '__main__':

	#run simulation
	nb_sensors = 3
	sensors = [TemperatureSensorSimulator("sensor"+str(i+1)) for i in range(0, nb_sensors)]
	[s.start() for s in sensors]

	#init socketio
	socketio.run( app, debug = True )
