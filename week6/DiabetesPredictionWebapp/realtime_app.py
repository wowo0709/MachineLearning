#!/usr/bin/env python
# coding: utf-8

# In[51]:


import numpy as np  # for manipulation
import pandas as pd  # for data loading

from sklearn.preprocessing import MinMaxScaler  # for scaling the attributes

import pickle  # for importing model

from flask import Flask, request, jsonify, render_template  # for handling web service

# Flask instantiation
app = Flask(__name__, template_folder='templates')

    
# class for data preprocessing
class data_preprocessing():
    def __init__(self, scaler):
        self.scaler = scaler
        
    def fit(self, X_train):
        # scale the features
        scaled_X_train = self.scaler.fit_transform(X_train)
        
        return scaled_X_train
        
    def transform(self, X_test):
        # scale the features
        scaled_X_test = self.scaler.transform(X_test)
        
        return scaled_X_test
    
    def savefittedobject(self):
        pickle.dump(self.scaler, open('diabetesscaler_retrain.pkl', 'wb'))
        
        
        
@app.route('/', methods=['GET', 'POST'])
def index():
    # model and fitted object loading
    model = pickle.load(open('diabetesregressionmodel.pkl', 'rb'))
    scaler = pickle.load(open('diabetesscaler.pkl', 'rb'))
    
    if request.method == 'GET':
        return(render_template('index.html'))
    if request.method == 'POST':
        # get input values
        # Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age
        pregnancies = int(request.form['Pregnancies'])
        glucose = int(request.form['Glucose'])
        bloodPressure = int(request.form['BloodPressure'])
        skinThickness = int(request.form['SkinThickness'])
        insulin = int(request.form['Insulin'])     
        bmi = float(request.form['BMI'])
        diabetesPedigreeFunction = float(request.form['DiabetesPedigreeFunction'])
        age = int(request.form['Age'])
        
        # convert input data to dataframe
        inputs_ = {'Pregnancies': pregnancies,
                  'Glucose': glucose,
                  'BloodPressure': bloodPressure,
                  'SkinThickness': skinThickness,
                  'Insulin': insulin,
                  'BMI': bmi,
                  'DiabetesPedigreeFunction': diabetesPedigreeFunction,
                  'Age': age
                  }
        
        inputs_df = pd.DataFrame.from_dict([inputs_])
        
        
        # preprocess the inputs
        preprocessing_ = data_preprocessing(scaler)
        inputs_preprocessed = preprocessing_.transform(inputs_df)
        
        # predict the price
        prediction = model.predict(inputs_preprocessed)
        
        # realtime training
        # preprocess the training data
        X_train = preprocessing_.fit(inputs_df)

        # retrain the model
        model.fit(X_train, [prediction[0]])

        # save the model
        pickle.dump(model, open('diabetesregressionmodel_retrain.pkl', 'wb'))

        # save the fitted objects
        preprocessing_.savefittedobject()
    
        return render_template('index.html', result=prediction[0])  
    
# running the application for serving
if __name__ == '__main__':
    app.run(host="224.194.7.101")

