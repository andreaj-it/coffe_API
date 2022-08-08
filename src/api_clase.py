from flask import Flask, render_template, jsonify, request 

app = Flask(__name__) # hacer ref al nombre del archivo

@app.route('/')
def hello_flask():
    return 'Hello Flask'

@app.route('/inicio')
def show_home():
    return render_template('Index.html')

@app.route('/url_variables/<string: name>/<int: age>')
def url_variables(name,age):
    if age < 18:
        return jsonify(message = 'Lo siento' + name + 'no estas autorizado'), 401
    else:
        return jsonify(message = 'Bienvenido' + name), 200 # por default es 200
    
if __name__ == '__main__':
    app.run(debug= True, host='127.0.0.1', port=5000) 