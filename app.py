from flask import Flask
from flask import render_template, request 
app = Flask(__name__)

import joblib
import os

loaded_model = joblib.load(os.getcwd()+"\\spam-detection-app\\stacked_model.joblib")
loaded_vectorizer = joblib.load(os.getcwd()+"\\spam-detection-app\\tokenizer.joblib") 

# Your niece has been arrested and needs $7,500.
# Hi, this is Cynde from HR. We have a couple question regarding your application. Please call 4732427394 to schedule a interview

@app.route("/")
def home():
    return render_template("index.html",prediction="")

@app.route('/predict', methods =["POST"])
def predict():
    if request.method == "POST":
        text= request.form.get("mail")
        print(text)
        X_train = loaded_vectorizer.transform([text])
        prediction = loaded_model.predict(X_train)
        return render_template("index.html",prediction="Spam" if prediction[0]==1 else "Ham")
       
if __name__ == "__main__":
    app.run(debug=True,port=8000)
