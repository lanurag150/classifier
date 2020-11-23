from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename

from gevent.pywsgi import WSGIServer
# Define a flask app
app = Flask(__name__)



# You can also use pretrained model from Keras
# Check https://keras.io/applications/
#from keras.applications.resnet50 import ResNet50
#model = ResNet50(weights='imagenet')
#print('Model loaded. Check http://127.0.0.1:5000/')




@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        dataset = pd.read_csv(file_path)
        x=dataset.iloc[:, [2,3]].values
        y=dataset.iloc[:, 4].values
        from sklearn.preprocessing import StandardScaler
        sc_x=StandardScaler()
        x=sc_x.fit_transform(x)
        from sklearn.model_selection import train_test_split
        x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0)

        from sklearn.ensemble import RandomForestClassifier
        classifier=RandomForestClassifier(n_estimators=10,criterion='entropy',random_state=0)
        classifier.fit(x_train,y_train)
        y_pred=classifier.predict(x_test)
        from sklearn.metrics import confusion_matrix
        cm1=confusion_matrix(y_test,y_pred)
        
        from sklearn.tree import DecisionTreeClassifier
        classifier=DecisionTreeClassifier(criterion='entropy',random_state=0)
        classifier.fit(x_train,y_train)
        y_pred=classifier.predict(x_test)
        from sklearn.metrics import confusion_matrix
        cm2=confusion_matrix(y_test,y_pred)
        
        from sklearn.naive_bayes import GaussianNB
        classifier=GaussianNB()
        classifier.fit(x_train,y_train)
        y_pred=classifier.predict(x_test)
        from sklearn.metrics import confusion_matrix
        cm3=confusion_matrix(y_test,y_pred)
        
        from sklearn.svm import SVC
        classifier=SVC(kernel='rbf', random_state=0)
        classifier.fit(x_train,y_train)
        y_pred=classifier.predict(x_test)
        from sklearn.metrics import confusion_matrix
        cm4=confusion_matrix(y_test,y_pred)
        
        from sklearn.svm import SVC
        classifier=SVC(kernel='linear',random_state=0)
        classifier.fit(x_train,y_train)
        y_pred=classifier.predict(x_test)
        from sklearn.metrics import confusion_matrix
        cm5=confusion_matrix(y_test,y_pred)
        
        from sklearn.neighbors import KNeighborsClassifier
        classifier=KNeighborsClassifier(n_neighbors=5,p=2)
        classifier.fit(x_train,y_train)
        y_pred=classifier.predict(x_test)
        from sklearn.metrics import confusion_matrix
        cm6=confusion_matrix(y_test,y_pred)
        
        from sklearn.linear_model import LogisticRegression
        lnclassifier=LogisticRegression(random_state=0)
        lnclassifier.fit(x_train,y_train)
        y_pred=lnclassifier.predict(x_test)
        from sklearn.metrics import confusion_matrix
        cm7=confusion_matrix(y_test,y_pred)
        
        acc1=(cm1[0][0]+cm1[1][1])/(cm1[0][0]+cm1[0][1]+cm1[1][0]+cm1[1][1])
        acc2=(cm2[0][0]+cm2[1][1])/(cm2[0][0]+cm2[0][1]+cm2[1][0]+cm2[1][1])
        acc3=(cm3[0][0]+cm3[1][1])/(cm3[0][0]+cm3[0][1]+cm3[1][0]+cm3[1][1])
        acc4=(cm4[0][0]+cm4[1][1])/(cm4[0][0]+cm4[0][1]+cm4[1][0]+cm4[1][1])
        acc5=(cm5[0][0]+cm5[1][1])/(cm5[0][0]+cm5[0][1]+cm5[1][0]+cm5[1][1])
        acc6=(cm6[0][0]+cm6[1][1])/(cm6[0][0]+cm6[0][1]+cm6[1][0]+cm6[1][1])
        acc7=(cm7[0][0]+cm7[1][1])/(cm7[0][0]+cm7[0][1]+cm7[1][0]+cm7[1][1])
        
        
        
        
        os.remove(file_path)#removes file from the server after prediction has been returned

        
       
        return "Accuracy RandomForest:"+str(acc7) + "Decision Tree:"+str(acc6) + "NaiveByes:"+str(acc5)  + "SVM:"+str(acc4) +"SVM :" +str(acc3) + "KNeighBour:" +str(acc2)+ "LogisticRegression:"+ str(acc1)
    return None

    #this section is used by gunicorn to serve the app on Heroku
if __name__ == '__main__':
        app.run()
    #uncomment this section to serve the app locally with gevent at:  http://localhost:5000
    # Serve the app with gevent 
    #http_server = WSGIServer(('', 5000), app)
    #http_server.serve_forever()

