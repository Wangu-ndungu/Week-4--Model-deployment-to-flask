# -*- coding: utf-8 -*-
"""
Created on Sun Jun 26 11:59:05 2022

@author: user
"""
import numpy as np
from flask import Flask, request, render_template, jsonify
import pickle


app = Flask(__name__, template_folder='template')
model = pickle.load(open('model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods= ['POST'])
def predict():
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    output = round(prediction[0])
           
    return render_template('index.html', prediction_text= 'Car price should be ${}'.format(output))
       
@app.route('/results',methods=['POST'])
def results():

    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)    
    
      
           
        
if __name__ == "__main__":
    app.run(port=5000, debug=True)
