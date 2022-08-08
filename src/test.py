#vamos a ver q corra el modelo
import pickle
import pandas as pd
import numpy as np

#el modelo recibe 8 datos para procesar y pronosticar

#country = 'Other'
#variety ='Other'
#aroma = 7.42
#aftertaste = 7.33
#acidity = 7.42
#body = 7.25
#balance = 7.33
#moisture = 0.0

country='Colombia'
variety='Caturra'
aroma = 7.83
aftertaste = 7.77
acidity = 7.33
body=7.67
balance=7.77
moisture=0.11

#tengo q suamr las cabeceras del objeto panda, el cuerpo son las variables q def mas arriba
cols = ['country_of_origin','variety','aroma','aftertaste','acidity','body','balance','moisture']

data = [country,variety,aroma,aftertaste,acidity,body,balance,moisture]

#armo el data frame
posted = pd.DataFrame(np.array(data).reshape(1,8), columns=cols) #fila con 8 columnas, lo q va a recibir

#ahora abrimos el archivo pickle para aplicar la prediccion
loaded_model = pickle.load(open('../models/coffee_model.pkl', 'rb')) #rb = read binario

result = loaded_model.predict(posted)

#me devuelve un archivo numpy, lo tengo q llevar a texto
text_result = result.tolist()[0]
print(text_result)
