from flask import Flask, render_template, jsonify, request 
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__) # hacer ref al nombre del archivo

@app.route('/')
def hello_flask():
    return 'Hello Flask'

@app.route('/inicio')
def show_home():
    return render_template('Index.html')

@app.route('/<string:country>/<string:variety>/<float:aroma>/<float:aftertaste>/<float:acidity>/<float:body>/<float:balance>/<float:moisture>/')

def result(country,variety,aroma,aftertaste,acidity,body,balance,moisture):
    cols = ['country_of_origin','variety','aroma','aftertaste','acidity','body','balance','moisture']
    data = [country,variety,aroma,aftertaste,acidity,body,balance,moisture]
    posted = pd.DataFrame(np.array(data).reshape(1,8), columns=cols) #fila con 8 columnas, lo q va a recibir
    loaded_model = pickle.load(open('../models/coffee_model.pkl', 'rb')) #rb = read binario
    result = loaded_model.predict(posted)
    text_result = result.tolist()[0]
    if text_result == 'Yes':
        return jsonify(message='Es un cafe de primera.'),200
    else : 
        return jsonify(message = 'No es cafe de primera.')


#@app.route('/url_variables/<string: name>/<int: age>')
#def url_variables(name,age):
#    if age < 18:
#        return jsonify(message = 'Lo siento' + name + 'no estas autorizado'), 401
#    else:
#        return jsonify(message = 'Bienvenido' + name), 200 # por default es 200
#https://developer.mozilla.org/en-US/docs/Web/HTTP/Status
#     
if __name__ == '__main__':
    app.run(debug= True, host='127.0.0.1', port=5000) 

#tengo q tener la opcion de ventanas emerentes para q levante,
#luego le paso por la url los valores tipo: urlxxx.io/Guatemala/Bourbon/7.83/7.77/7.33/7.67/7.77/0.11

#tenemos q montar un server web en heroku, web: gunicorn app:app
#tengo q montar mi estructura en heroku