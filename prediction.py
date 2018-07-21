from __future__ import print_function
import sys
import flask
from flask import jsonify
from flask import request
# import numpy as np
# import pandas as pd
# from sklearn.metrics import classification_report
# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import accuracy_score
# from sklearn.model_selection import train_test_split
# from sklearn.model_selection import cross_val_score
# from sklearn.model_selection import KFold
# from sklearn.preprocessing import StandardScaler
# from sklearn.svm import SVC
# from sklearn.linear_model import LogisticRegression
# import time
# import mysql.connector as sql

app = flask.Flask(__name__)

# train the system on startup.
# make subsequent predictions with input test data.

def test_model(data):
    print("Testing model against input...")
    return jsonify(prediction='Benign')
    print("Prediction: ")

@app.before_request
def train_model():
    print("Training Model...")
    # db_connection = sql.connect(host='http://localhost:3306/', database='health', user='root', password='')
    # data = pd.read_sql('SELECT * FROM health.uw_data', con=db_connection)
    # Y = data['diagnosis'].values
    # X = data.drop('diagnosis', axis=1).values

    # # Below shows SVC is better than Logistics model; DO NOT NEED FOR PRODUCTION!
    # num_folds = 10
    # kfold = KFold(n_splits=num_folds, random_state=123)
    # start = time.time()
    # cv_results = cross_val_score(LogisticRegression(), X, Y, cv=kfold, scoring='accuracy')
    # end = time.time()
    # print( "Logistics regression accuracy: %f, run time: %f)" % (cv_results.mean(), end-start))

    # start = time.time()
    # scaler = StandardScaler().fit(X)
    # X_scaled = scaler.transform(X)
    # cv_results = cross_val_score(SVC(C=2.0, kernel="rbf"), X_scaled, Y, cv=kfold, scoring='accuracy')
    # end = time.time()
    # print( "SVC accuracy: %f, run time: %f)" % (cv_results.mean(), end-start))

    # # THIS IS WHAT IS NEEDED FOR PRODUCTION TO ESTIMATE THE MODEL
    # scaler = StandardScaler().fit(X)
    # X_scaled = scaler.transform(X)
    # model = SVC(C=2.0, kernel='rbf')
    # model.fit(X_scaled, Y)


    # # THIS IS WHAT IS NEEDED FOR PREDICTION
    # X_test_scaled = scaler.transform(X[10,:].reshape(1,-1))
    # predictions = model.predict(X_test_scaled)

    # print("Predicted: {}, actual: {}".format(predictions[0], Y[10]))
    # print("Model trained.")

@app.route('/prediction', methods=['POST'])
def prediction():
    data = request.get_json(silent=True)
    return test_model(data)

if __name__ == "__main__":
    print("Starting Flask service")
    app.run()
