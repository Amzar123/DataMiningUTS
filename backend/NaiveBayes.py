from flask import Flask, request, render_template
from sklearn.metrics import accuracy_score
from sklearn import preprocessing,naive_bayes, metrics
import pickle
import pandas as pd 
import numpy as np 

app = Flask(__name__,template_folder="../templates")

model_file = open('../model/playTennis.pkl', 'rb')
model = pickle.load(model_file, encoding='bytes')

@app.route('/')
def index():
    # return "success"
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    '''
    Predict the insurance cost based on user inputs
    and render the result to the html page
    '''
    outlook=float(request.form['outlook'])
    
    temp=float(request.form['temp'])

    humidity=float(request.form['humidity'])

    wind=float(request.form['wind'])

    day=float(request.form['day'])

    X=np.array([[day,outlook,temp,humidity,wind]])
    
    prediction = model.predict(X)

    output = round(prediction[0],0)
    if (output==0):
        kelas="Tidak Bermain"
    elif (output==1):
        kelas="Bermain"

    return render_template('index.html',kelas=kelas, outlook=outlook, temprature=temp, humidity=humidity, wind=wind)


if __name__ == '__main__':
    app.run(debug=True)