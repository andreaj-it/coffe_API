from flask import Flask, render_template, jsonify, request 
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)
model = pickle.load(open('coffee_model.pkl', 'rb'))

@app.route('/')

def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])

def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Como es el cafe? {} (Yes=Es un cafe de primera, No=No es cafe de primera)'.format(output))

"""
@app.route('/<string:country>/<string:variety>/<float:aroma>/<float:aftertaste>/<float:acidity>/<float:body>/<float:balance>/<float:moisture>/')
def result(country,variety,aroma,aftertaste,acidity,body,balance,moisture):
    cols = ['country_of_origin','variety','aroma','aftertaste','acidity','body','balance','moisture']
    data = [country,variety,aroma,aftertaste,acidity,body,balance,moisture]
    posted = pd.DataFrame(np.array(data).reshape(1,8), columns=cols) #fila con 8 columnas, lo q va a recibir
    loaded_model = pickle.load(open('coffee_model.pkl', 'rb')) #rb = read binario
    result = loaded_model.predict(posted)
    text_result = result.tolist()[0]
    if text_result == 'Yes':
        return jsonify(message='Es un cafe de primera.'),200
    else : 
        return jsonify(message = 'No es cafe de primera.')
"""

if __name__=="__main__":
    app.run(port=5000, debug=True)