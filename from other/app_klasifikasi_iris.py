from flask import Flask, request, render_template
import pickle
import pandas as pd 
import numpy as np 

app = Flask(__name__)

model_file = open('playTennis.pkl', 'rb')
model = pickle.load(model_file, encoding='bytes')

@app.route('/')
def index():
    return render_template('index.html', hasil=0)

@app.route('/predict', methods=['POST'])
def predict():
    '''
    Predict the insurance cost based on user inputs
    and render the result to the html page
    '''
    day=float(request.form['day'])
    
    outlook=float(request.form['outlook'])
    
    temperature=float(request.form['temperature'])

    humidity=float(request.form['humidity'])

    wind=float(request.form['wind'])

    x=np.array([[day,outlook,temperature,humidity,wind]])

 
    
    prediction = model.predict(x)
    output = round(prediction[0],0)
    if (output==0):
        kelas="No"
    else:
        kelas="Play"


    return render_template('index.html', hasil=kelas, day=day,outlook=outlook, temperature=temperature, humidity=humidity, wind=wind)


if __name__ == '__main__':
    app.run(debug=True)