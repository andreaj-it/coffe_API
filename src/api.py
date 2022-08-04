import numpy as np
import pickle

from flask import Flask, request, render_template
from flask import jsonify

def create_app(enviroment):
    app = Flask(__name__)
    return app

def get_users():
    response = {'message': 'success'}
    return jsonify(response)

def get_users():
    response = {'message': 'success'}
    return jsonify(response)

model = pickle.load(open('../models/titanic_model.pkl', 'rb'))

app = create_app()
app.route('/api/v1/users', methods=['GET'])






from flask import Flask
from flask import jsonify
from flask import Flask, jsonify
from flask_restful import Api, Resource, reqparse
import pickle
import numpy as np
import json

import numpy as np
from flask import Flask, request, render_template
import pickle

#usando ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
#[5,58,1,2,3]

app = Flask(__name__)
model = pickle.load(open('../models/titanic_model.pickle', 'rb'))

model = LogisticRegression()
model.fit(X, y)
# define one new instance
Xnew = [[-0.79415228, 2.10495117]]
# make a prediction
ynew = model.predict(Xnew)
print("X=%s, Predicted=%s" % (Xnew[0], ynew[0]))


nt_features = [int(x) for x in request.form.values()]
final_features = [np.array(int_features)]
prediction = model.predict(final_features)
input_data = (3,0,35,0,0,8.05,0)
input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
prediction = model.predict(input_data_reshaped)
#print(prediction)
