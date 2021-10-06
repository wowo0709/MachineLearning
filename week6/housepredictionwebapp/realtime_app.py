#!/usr/bin/env python
# coding: utf-8

# In[51]:


import numpy as np  # for manipulation
import pandas as pd  # for data loading

from sklearn.preprocessing import StandardScaler  # for scaling the attributes
from sklearn.preprocessing import OneHotEncoder  # for handling categorical features
from sklearn.impute import SimpleImputer   # for handling missing data

import pickle  # for importing model

from flask import Flask, request, jsonify, render_template  # for handling web service

# Flask instantiation
app = Flask(__name__, template_folder='templates')

# Custom class for combined attributes
class CombinedAttributesAdder():    
    def fit(self, X, y=None):
        return self
    def transform(self, X, rooms_ix, bedrooms_ix, population_ix, households_ix):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
        
        X = np.delete(X, [households_ix, rooms_ix, population_ix, bedrooms_ix], 1)
        
        return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
    
# class for data preprocessing
class data_preprocessing():
    def __init__(self, imputer, scaler, ohencoder):
        self.imputer = imputer
        self.attr_add = CombinedAttributesAdder()
        self.stdscale = scaler
        self.ohe = ohencoder
        
    def fit(self, X, rooms_ix, bedrooms_ix, population_ix, households_ix): 
        # fit and transform the training data
        house_num = X.drop("ocean_proximity", axis=1)
        house_cat = X[["ocean_proximity"]]
        
        # handle missing data
        self.imputer.fit(house_num)
        X_train_imp = self.imputer.transform(house_num)
        X_train_imp = pd.DataFrame(X_train_imp, columns=house_num.columns, index=X.index)
        
        # combined attributes
        housing_addtl_attr = self.attr_add.transform(X_train_imp.values, rooms_ix, 
                                                     bedrooms_ix, population_ix, households_ix)
        
        # scale the features
        self.stdscale.fit(housing_addtl_attr)
        X_train_imp_scaled = self.stdscale.transform(housing_addtl_attr)
        
        # handle categorical input feature
        self.ohe.fit(house_cat)
        X_train_ohe = self.ohe.transform(house_cat)
        
        # concatenate features
        X_train = np.concatenate([X_train_imp_scaled, X_train_ohe], axis=1)
        
        return X_train
        
    def transform(self, X, rooms_ix, bedrooms_ix, population_ix, households_ix): 
        # transform the test data (use the fitted imputer, 
        #                         standardscaler, onehotencoder, 
        #                         combinedattribute from training)
        house_num = X.drop("ocean_proximity", axis=1)
        house_cat = X[["ocean_proximity"]]
        
        # handle missing data
        X_test_imp = self.imputer.transform(house_num)
        X_test_imp = pd.DataFrame(X_test_imp, columns=house_num.columns, index=X.index)
        
        # combined attributes
        housing_addtl_attr = self.attr_add.transform(X_test_imp.values, rooms_ix, 
                                                     bedrooms_ix, population_ix, households_ix)
        
        # scale the features
        X_test_imp_scaled = self.stdscale.transform(housing_addtl_attr)
        
        # handle categorical input feature
        X_test_ohe = self.ohe.transform(house_cat)
        
        # concatenate features
        X_test = np.concatenate([X_test_imp_scaled, X_test_ohe], axis=1)
        
        return X_test
    
    def savefittedobject(self):
        pickle.dump(self.imputer, open('houseimputer_retrain.pkl', 'wb'))
        pickle.dump(self.stdscale, open('housescaler_retrain.pkl', 'wb'))
        pickle.dump(self.ohe, open('houseohencoder_retrain.pkl', 'wb'))
        
@app.route('/', methods=['GET', 'POST'])
def index():
    # model and fitted object loading
    model = pickle.load(open('houseregressionmodel.pkl', 'rb'))
    imputer = pickle.load(open('houseimputer.pkl', 'rb'))
    scaler = pickle.load(open('housescaler.pkl', 'rb'))
    ohencoder = pickle.load(open('houseohencoder.pkl', 'rb'))
    
    if request.method == 'GET':
        return(render_template('index.html'))
    if request.method == 'POST':
        # get input values
        longitude = float(request.form['longitude'])
        latitude = float(request.form['latitude'])
        housingmedianage = float(request.form['housingmedianage'])
        totalrooms = float(request.form['totalrooms'])
        totalbedrooms = request.form['totalbedrooms']        
        population = float(request.form['population'])
        households = float(request.form['households'])
        medianincome = float(request.form['medianincome'])
        oceanproximity = request.form['oceanproximity']
        
        # handle missing input in total_bedrooms attribute
        if totalbedrooms == '':
            totalbedrooms = float('nan')
        else:
            totalbedrooms = float(totalbedrooms)
        
        # new category creation by assuming median income is a very important attribute 
        income_cat = pd.cut([medianincome],
                              bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                              labels=[1, 2, 3, 4, 5])
        
        # convert input data to dataframe
        inputs_ = {'longitude': longitude,
                  'latitude': latitude,
                  'housing_median_age': housingmedianage,
                  'total_rooms': totalrooms,
                  'total_bedrooms': totalbedrooms,
                  'population': population,
                  'households': households,
                  'median_income': medianincome,
                  'ocean_proximity': oceanproximity,
                  'income_cat': income_cat}
        
        inputs_df = pd.DataFrame(inputs_)
        
        # get the column indices to be used in getting additional attributes
        col_names = ["total_rooms", "total_bedrooms", "population", "households"]
        rooms_ix, bedrooms_ix, population_ix, households_ix = [
            inputs_df.columns.get_loc(c) for c in col_names] # get the column indices
        
        # preprocess the inputs
        preprocessing_ = data_preprocessing(imputer, scaler, ohencoder)
        inputs_preprocessed = preprocessing_.transform(inputs_df, rooms_ix, bedrooms_ix, 
                                                       population_ix, households_ix)
        
        # predict the price
        prediction = model.predict(inputs_preprocessed)
        
        # realtime training
        # preprocess the training data
        data_X_train = preprocessing_.fit(inputs_df, rooms_ix, bedrooms_ix, 
                                                   population_ix, households_ix)

        # retrain the model
        model.fit(data_X_train, [prediction[0]])

        # save the model
        pickle.dump(model, open('houseregressionmodel_retrain.pkl', 'wb'))

        # save the fitted objects
        preprocessing_.savefittedobject()
    
        return render_template('index.html', result=prediction[0])  
    
# running the application for serving
if __name__ == '__main__':
    app.run(host="128.134.65.180")

