from __future__ import print_function
import sys
import flask
from flask import jsonify
from flask import request
from flask import g

import numpy as np
import pandas as pd
import pickle
import time
import mysql.connector as sql

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

app = flask.Flask(__name__)

def _train_logistic_regression(X, Y):
    num_folds = 10
    kfold = KFold(n_splits=num_folds, random_state=123)
    return cross_val_score(LogisticRegression(), X, Y.values.ravel(), cv=kfold, scoring='accuracy')
    
def _train_svc(X, Y):
    num_folds = 10
    kfold = KFold(n_splits=num_folds, random_state=123)
    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)
    return cross_val_score(SVC(C=2.0, kernel="rbf"), X_scaled, Y.values.ravel(), cv=kfold, scoring='accuracy')

@app.route('/train/SVC', methods=['GET'])
def train_svc_model():
    print("SVC is being trained on new dataset...")
    try:
        db_connection = sql.connect(host='localhost', database='health', user='root', password='')
        X = pd.read_sql('SELECT * FROM health.uw_data', con=db_connection).drop({'id', 'diagnosis'}, axis=1).values
        Y = pd.read_sql('SELECT diagnosis FROM health.uw_data', con=db_connection)
        _training_upsert(db_connection, "SVC", float(_train_svc(X, Y).mean()), len(X))
        return jsonify(response="Success")
    except Exception as e:
        return jsonify(response=e.args)
    finally:
        db_connection.close();

@app.route('/train/LogisticRegression', methods=['GET'])
def train_logistic_regression_model():
    print("Logistic Regression is being trained on new dataset...")
    try:
        db_connection = sql.connect(host='localhost', database='health', user='root', password='')
        X = pd.read_sql('SELECT * FROM health.uw_data', con=db_connection).drop({'id', 'diagnosis'}, axis=1).values
        Y = pd.read_sql('SELECT diagnosis FROM health.uw_data', con=db_connection)
        _training_upsert(db_connection, "LogisticRegression", float(_train_logistic_regression(X, Y).mean()), len(X))
        return jsonify(response="Success")
    except Exception as e:
        return jsonify(response=e.args)
    finally:
        db_connection.close();

def _training_upsert(db_connection, modelName, cvAccuracy, numberOfRecords):
    query = """
        UPDATE models
        SET cv_accuracy=%s, model_entry_timestamp=CURRENT_TIMESTAMP, number_of_records=%s
        WHERE model_name=%s
        """;
    data = (cvAccuracy, numberOfRecords, modelName)
    try:
        cursor = db_connection.cursor()
        cursor.execute(query, data)
        db_connection.commit()
    except Exception as e:
        return jsonify(response=e)
    finally:
        cursor.close()

@app.route('/test/LogisticRegression', methods=['GET'])
def test_logistic_regression():
    data = request.args.get('patientHistoryId')
    # model.predict on data.
    return jsonify(prediction="B")

@app.route('/test/LogisticRegression', methods=['GET'])
def test_svc():
    data = request.args.get('patientHistoryId')
    # model.predict on data.
    return jsonify(prediction="B")

@app.route('/test', methods=['GET'])
def test():
    # Stubbed
    return jsonify(prediction="B")

if __name__ == "__main__":
    print("Starting Flask service")
    db_connection = sql.connect(host='localhost', database='health', user='root', password='')
    cursor = db_connection.cursor()
    app.run()


# restored_model = pickle.loads(pickled_model)
# X_test_scaled = scaler.transform(X[10, :].reshape(1, -1))
# predictions = restored_model.predict(X_test_scaled)
# probability = restored_model.predict_proba(X_test_scaled)

# print("Predicted: {}, with probability: {}, actual: {}".format(
#     predictions[0], probability[0], Y[10]))