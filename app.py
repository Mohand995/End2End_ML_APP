import pickle 
import os
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd

app=Flask(__name__)
model=pickle.load(open("reg_model_2.pkl",'rb'))
scaler=pickle.load(open("Scaler.pkl",'rb'))
@app.route("/")
def home():
    return render_template('home.html')

@app.route("/predict_api",methods=['POST'])
def predict_api():
    data=request.json['data']
    print(np.array(list(data.values())).reshape(1,-1))
    data_scaled=scaler.transform(np.array(list(data.values())).reshape(1,-1))
    output=model.predict(data_scaled)
    return jsonify(output[0])

@app.route("/predict",methods=['POST'])
def predict():
    data=[float(x) for x in request.form.values()]
    final_input=scaler.transform(np.array(data).reshape(1,-1))
    print(final_input)
    output=model.predict(final_input)[0]
    return render_template("home.html",prediction_text="the prediction price is {}".format(output))

    
if __name__=="__main__":
    app.run(debug=True)
