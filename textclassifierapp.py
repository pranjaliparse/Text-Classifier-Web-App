from flask import Flask, render_template, url_for, request
from flask_material import Material

import pandas as pd
import numpy as numpy
import pickle

from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
#count_vect = CountVectorizer()

#create an instance of the Flask class
app = Flask(__name__)
#place holder for current module
Material(app)

@app.route("/")
def index():
    return render_template('indextextc.html')

@app.route('/', methods=['POST'])
def analyze():
    if request.method == 'POST':
        #POST request method requests that a web server accepts the data enclosed in the body of the request message
        #Most likely for storing it
        MyText = request.form['MyText']
        sample_data=[MyText]
        print("hey",MyText)
        # Clean the data by converting from Unicode to Float
        with open('mnb_textclassification_pickle','rb') as f:
            clf=pickle.load(f)
        with open('count_textclassification_pickle','rb') as f1:
            count_vect=pickle.load(f1)
        result_prediction=clf.predict(count_vect.transform(sample_data))

        
    return render_template('indextextc.html', 
    MyText=MyText,
    result_prediction=result_prediction)


if __name__=='__main__':
    app.run(debug=True)