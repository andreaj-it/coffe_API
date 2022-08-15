from flask import Flask, render_template, jsonify, request 
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__,template_folder='templates')

@app.route('/')

def home():
    return render_template('index.html')

@app.route('/', methods=['POST'])

def predict():
    if request.method == 'POST':
        country = request.form['country']
        variety = request.form['variety']
        aroma = request.form['aroma']
        aftertaste = request.form['aftertaste']
        acidity = request.form['acidity']
        body = request.form['body']
        balance = request.form['balance']
        moisture = request.form['moisture']

        cols = ['country_of_origin','variety','aroma','aftertaste','acidity','body','balance','moisture']
        data = [country,variety,aroma,aftertaste,acidity,body,balance,moisture]
        to_predict_list = pd.DataFrame(np.array(data).reshape(1,8), columns=cols) #fila con 8 columnas, lo q va a recibir

        #to_predict_list = request.form.to_dict()
        #to_predict_list=list(to_predict_list.values())
        #to_predict_list = list(map(int, to_predict_list))
        
        result = ValuePredictor(to_predict_list)
        
        if result == 'Yes':
            prediction='Es un cafe de primera'
        else:
            prediction='NO es un cafe de primera'
            
        return render_template("index.html",prediction=prediction)

def ValuePredictor(to_predict_list):
    to_predict = to_predict_list
    #to_predict = np.array(to_predict_list).reshape(1,8)
    loaded_model = pickle.load(open("coffee_model.pkl","rb"))
    #model = pickle.load(open('coffee_model.pkl', 'rb'))
    result = loaded_model.predict(to_predict)
    return result[0]

"""
example = ['Other', 'Other', 7.42, 7.33, 7.42, 7.25, 7.33, 0]
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