# MicroSec Machine Learning Challenge Internship

## Planning
* Wait for 00h to receive the question mail
* 00h15   -   00h30   : Start to read questions so i can think about it during the night
* 8h15    -   10h30   : part1 understand check
* 10h30   -   11h     : part1 redaction (schema)
* 11h00   -   13h00   : part2 understand check, problem/data, compute data to prepare training
* 14h30   -   17h30   : part2 creating models using keras (deep learning) and using scikit learn (machine learning)
* 17h30   -   18h30   : Write reports

## Part 1 : Django/Flask Programming
1) I did'nt know anything about Celery or Websockets so i started to learn how it works

2) Then i tried to find similar projects on github
I found a python chat app using websocket : https://github.com/rakibtg/Python-Chat-App
I learnt how it works then get think about my data structure => Using Database/Sensors simulation

3) Then i created my simulation with "Fake SQL Database" and "Fake Sensors" because i did'nt have enough time to create/implement a full sql database for this daily project (i know how works a SQL database...)

4) Then i implemented my dashboard based on python chat app using flask
The picture structure.png in part1/ show the structure of the app

Some python3 modules needs to be installed :
### Installation
`pip3 install flask flask-socketio eventlet`
### Run
Simply go in part1/
`python3 app.py`

Then connect to http://localhost:5000/ and it will start the simulation.
"Fake" sensors will send data to api and update client website automatically as shown on the structure.png

## Part 2 : NILM Data Science
1) I started to look at the data
`python3 datavisualization.py`

I created graphics to have effective current in function of time for pc1, pc2, sensor and main data (checkout part2_keras/images/)
Then i thought about my data structure and models but i wasn't enough confident on my idea and chose to do some internet searches.

2) Then i started again to search similar project on github looking for "state of the art" projects.
I found https://github.com/Vibhuti-B/NILM which also helped me to understand data.

3) I was more confident on my initial idea and i started to implement my first idea. It seems like my main idea was almost the same as the github implementation that's why you will find some similarities.
`python3 pretrain.py`
To compute data and store them in data/ so we don't need to compute it each time we train.

4) **(part2_keras/)** I then chose my intelligent framework. I started with deep learning using keras even if there isn't a lot of data. I used keras to simply implement my models.
I created two models for PCs : very simple push forward neural networks with onehot encoding output to have classification problem. Almost the same thing for temperature sensor but i had to modify the network model because this is a binary classification on the contrary of PCs network which is categories classification.

#### Accuracy
`pc1_model accuracy 75,6%`

`pc2_model accuracy 76,2%`

`sensor_model accuracy 64,8%`

I was not very satisfied by accuracy of my models that's why i chose to implement machine learning too.

5) **(part2_sklearn/)** So, based on keras works i implemented intelligent algorithms using scikit learn.
Based on Vibhuti-B works. I tried lot of different algorithms and found similar algorithms for my 3 models.

#### Accuracy
`pc1_model accuracy 87.8%`

`pc2_model accuracy 76.2%`

`sensor_model accuracy 75,9%`

Accuracy looks pretty similar so the lack of accuracy may be due to the lack of data.

For future uses, i would choose the deep learning option because accuracy will increase with numbers of accumulated data. Machine Learning algorithms will not improve with more data.

### Installation
`pip3 install sklearn tensorflow keras matplotlib`
### Run
Simply go in part2_keras/ or part2_sklearn/

#### Precompute data
(Uloaded models are already computed


`python3 pretrain.py`

#### Train
Uploaded models are already trained


`python3 train.py`

#### Test


`python3 test.py`


Tests will output a csv file "outut.csv" in the current folder with predictions.

### References
* https://github.com/rakibtg/Python-Chat-App
* https://github.com/Vibhuti-B/NILM

## Authors

* **Théo Guidoux** - [zetechmoy](https://github.com/zetechmoy) - [@TGuidoux](https://twitter.com/TGuidoux)
