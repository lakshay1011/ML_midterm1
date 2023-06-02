import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle
app = Flask(__name__) #template_folder='template')
model = pickle.load(open('Naive123.pkl','rb'))

# Extracting features of dataset
dataset = pd.read_csv('Set-3.csv')
X = dataset.iloc[:, 0:8].values
y = dataset.iloc[:,8].values

#Missing Data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values= np.NAN, strategy= 'mean', copy=True)
imputer = imputer.fit(X[:, 2:7]) 
X[:, 2:7]= imputer.transform(X[:, 2:7])

#encoding categorical data
from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
# Applying PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = 3)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_


@app.route('/')
def home():
  
    return render_template("hcindex.html")
@app.route('/predict',methods=['GET'])
def predict():
  '''
  For rendering results on HTML GUI
  '''
  gender = int(request.args.get('gender'))
  glucose = int(request.args.get('glucose'))
  bp = int(request.args.get('bp'))
  skin = int(request.args.get('skin')) 
  insulin = int(request.args.get('insulin'))
  bmi = int(request.args.get('bmi')) 
  pf = int(request.args.get('pf'))
  age = int(request.args.get('age')) 
  
  arr=[[gender,glucose,bp,skin,insulin,bmi,pf,age]]
 

  arr1 = pca.transform(arr)
  predict = model.predict(arr1)
  return render_template('hcindex.html', prediction_text='Model  has predicted  : {}'.format(predict))
if __name__ =='__main__':
 app.run()
