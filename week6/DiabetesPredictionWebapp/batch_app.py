#!/usr/bin/env python
# coding: utf-8

# In[95]:


import numpy as np  # for manipulation
import pandas as pd  # for data loading

from sklearn.preprocessing import MinmaxScaler  # for scaling the attributes

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
        scaled_X_train = self.stdscale.fit_transform(X_train)
        
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
    
    # load the dataset
    diabetes = pd.read_csv('diabetes_data.csv')
    
    if request.method == 'GET':
        return(render_template('index.html'))
    if request.method == 'POST':
        # get input values
        # Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age
        pregnancies = float(request.form['Pregnancies'])
        glucose = float(request.form['Glucose'])
        bloodPressure = float(request.form['BloodPressure'])
        skinThickness = float(request.form['SkinThickness'])
        insulin = request.form['Insulin']        
        bmi = float(request.form['BMI'])
        diabetesPedigreeFunction = float(request.form['DiabetesPedigreeFunction'])
        age = float(request.form['Age'])
        
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
        
        inputs_df = pd.DataFrame(inputs_)
        
        
        # preprocess the inputs
        preprocessing_ = data_preprocessing(scaler)
        inputs_preprocessed = preprocessing_.transform(inputs_df)
        
        # predict the price
        prediction = model.predict(inputs_preprocessed)
        
        # batch training
        # adding the Outcome in the data for retraining
        inputs_df['Outcome'] = int(prediction[0])
        # saving to csv the new data
        inputs_df.to_csv('diabetes_data.csv', mode='a', index=False, header=False)
        # retraining
        if len(diabetes) > 40:  
            # assign the training data
            X_train = diabetes.drop("Outcome", axis=1)
            y_train = diabetes["Outcome"].copy()
            
            # preprocess the training data
            preprocessed_X_train = preprocessing_.fit(X_train)
            
            # retrain the model
            model.fit(preprocessed_X_train, y_train)
            
            # save the model
            pickle.dump(model, open('diabetesregressionmodel_retrain.pkl', 'wb'))
            
            # save the fitted objects
            preprocessing_.savefittedobject()
    
        return render_template('index.html', result=prediction[0])  
    
# running the application for serving
if __name__ == '__main__':
    app.run(host="223.194.7.101")


# In[91]:





# In[92]:





# In[93]:





# In[ ]:




