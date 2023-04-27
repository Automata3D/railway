import joblib
import json
import pickle
import pandas as pd
from flask import Flask, jsonify, request
from peewee import (
    SqliteDatabase, PostgresqlDatabase, Model, IntegerField,
    FloatField, TextField, IntegrityError,BooleanField
)
from playhouse.shortcuts import model_to_dict
from playhouse.db_url import connect
import os
from uuid import uuid4
import reverse_geocoder as rg

# The connect function checks if there is a DATABASE_URL env var.
# If it exists, it uses it to connect to a remote postgres db.
# Otherwise, it connects to a local sqlite db stored in the predictions.db file.
DB = connect(os.environ.get('DATABASE_URL') or 'sqlite:///predictions.db')


class Prediction(Model):
    observation_id = IntegerField(unique=True)
    observation = TextField()
    proba = FloatField()
    true_class = IntegerField(null=True)
    outcome = BooleanField(null=True)
    predicted_outcome = BooleanField(null=True)
    
    class Meta:
        database = DB


DB.create_tables([Prediction], safe=True)

# End database stuff
########################################

########################################
# Unpickle the previously-trained model


with open('columns.json') as fh:
    columns = json.load(fh)

pipeline = joblib.load('pipeline.pickle')

with open('dtypes.pickle', 'rb') as fh:
    dtypes = pickle.load(fh)


# End model un-pickling
########################################


########################################
# Begin webserver stuff

app = Flask(__name__)


@app.route('/should_search/', methods=['POST'])
def should_search():
    # Flask provides a deserialization convenience function called
    # get_json that will work if the mimetype is application/json.
    obs_dict = request.get_json()

    ################################################
    # Missing `observation_id` produces an error
    ################################################
    response = {}
    print(columns)
    
    # Missing observation id 
    try:
        request_id = obs_dict['observation_id']
    except:
        request_id = None
        return ({"observation_id": request_id,
                "error": "observation_id is missinggg"})
    
    request_data = obs_dict
    
    
    
    
    
    ################################################
    # Missing columns produce an error
    ################################################
    #if len(request_data) < len(columns):
    #    response = {"observation_id": request_id,
    #                "error": "there's data missing"}
    #    return response
        
    ################################################
    # Test extra columns produce an error
    ################################################
    #if len(request_data) > len(columns):
    #    response = {"observation_id": request_id,
    #                "error": "there's extra data"}
    #    return response
    df = pd.DataFrame(request_data, index=[0])
    df['Date'] = pd.to_datetime(df['Date'])
    df['Date'] = df['Date'].dt.date
    df = df.set_index('Date')
    df.index = pd.to_datetime(df.index)
    df['day_of_week'] = df.index.weekday
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['day'] = df.index.day
    
    # create a list of tuples containing the latitude and longitude coordinates
    coordinates = [(lat, lon) for lat, lon in zip(df['Latitude'], df['Longitude'])]
    # use reverse geocoding to get the nearest neighborhood for each set of coordinates
    results = rg.search(coordinates)
    # create a new column in the dataframe to store the neighborhood names
    df['Neighborhood'] = [result['name'] for result in results]
    df = df.drop(columns={'observation_id'})
    # Assuming you already have the DataFrame `df`
    request_data = df.iloc[0].to_dict()
    # Removing keys with missing values ("nan")
    request_data = {k: v for k, v in request_data.items() if v != "nan"}
    # Now do what we already learned in the notebooks about how to transform
    # a single observation into a dataframe that will work with a pipeline.
    request_df = pd.DataFrame([request_data], columns = columns).astype(dtypes)
    
    # Now get ourselves an actual prediction of the positive class.
    #proba = pipeline.predict_proba(request_df)[0, 1]
    



    
    
    
    
    outcome = pipeline.predict(request_df)
    #outcome_proba = pipeline.predict_proba(request_df)[0]
    prediction = pipeline.predict(request_df)[0]
    #response = request_df.to_dict('records')[0]
    #response['readmitted'] = outcome.tolist()
    proba = pipeline.predict_proba(request_df)[0, 1]
    #response = {'proba': proba,'outcome':int(outcome[0])}
    #response['outcome'] = int(outcome[0])
    #response = {'prediction': bool(prediction), 'proba': proba}
    response = {'outcome': bool(prediction)}
    
    p = Prediction(
        observation_id=request_id,
        proba=proba,
        observation=request_data,
        predicted_outcome=bool(prediction),
        outcome = ""
    )
    try:
        p.save()
    except IntegrityError:
        error_msg = "ERROR: Observation ID: '{}' already exists".format(request_id)
        response["error"] = error_msg
        print(error_msg)
        DB.rollback()
    return jsonify(response)


#######################################################################

@app.route('/search_result/', methods=['POST'])
def search_result():
    
    obs = request.get_json()
    response = {}
    print(json.dumps(obs, indent=4))
    try:
        p = Prediction.get(Prediction.observation_id == obs['observation_id'])
        print(Prediction.get(Prediction.observation_id == obs['observation_id']))
        p.outcome = bool(obs['outcome'])
        #p.true_class = obs['readmitted']
        p.save()
        response['observation_id'] = p.observation_id
        response['outcome'] = p.outcome
        response['predicted_outcome'] = p.predicted_outcome
        return jsonify(response)
    except Prediction.DoesNotExist:
        error_msg = 'Observation ID: "{}" does not exist'.format(obs['observation_id'])
        return jsonify({'error': error_msg})
    
    #response = {'observation_id':p.observation_id,'outcome': p.outcome, 'predicted_outcome':p.predicted_outcome}
    
 
    
    
    

@app.route('/update/', methods=['POST'])
def update():
    obs = request.get_json()
    try:
        p = Prediction.get(Prediction.observation_id == obs['observation_id'])
        p.true_class = obs['readmitted']
        p.save()
        return jsonify(model_to_dict(p))
    except Prediction.DoesNotExist:
        error_msg = 'Observation ID: "{}" does not exist'.format(obs['observation_id'])
        return jsonify({'error': error_msg})
    
# End webserver stuff
########################################

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True, port=5000)
