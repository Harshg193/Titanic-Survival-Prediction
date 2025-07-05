# Analyzze Titanic Dataset and prepare a machine learning model to predict the passenger surivival

#Ml Algo
#1 Logistic Reg
#2 Support vector Machine
#3 Naive Bais
#4 K-Nearst Neighbour


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

#import a lib for label encoding
from sklearn.preprocessing import LabelEncoder

#import lib for train test set
from sklearn.model_selection import train_test_split
#import lib for accuracy score
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix

df = pd.read_csv('Titanic-Dataset.csv')

#Basic Analysis of Data
df.head()

df.shape

#Basic info about the datatype of each colunn
df.info()

# Check the missing value in our dataset
df.isna().sum()

df["Pclass"].value_counts()

# Dividing the dataframe into dependent and independent vari
X = df[["Pclass","Age","Sex","SibSp","Parch","Fare"]]
Y = df["Survived"]

# Checking the info on X
X.info()

#
X['Age']=X["Age"].fillna(X["Age"].mean())

X.head()

Y.head()

encoder = LabelEncoder()
X["Sex"] = encoder.fit_transform(X["Sex"])

X.head()

#dividing dataset into traing and test set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2,random_state = 1)

# Dictionary of Model for Model Setup (to compare model)
models = {
    "Logistic Regression" : LogisticRegression(),
    "Support Vector Machine" : SVC(),
    "Naive Bais" : GaussianNB(),
    "K-Nearst Neighbour" : KNeighborsClassifier(),
    "Decision Tree" : DecisionTreeClassifier(),
    "Random Forest" : RandomForestClassifier()
}

#Training and evaluting the model
results = []
for name, model in models.items():
  model.fit(X_train, Y_train) #tain my model
  Y_pred = model.predict(X_test)

  #classification report of all ml algo
  print(f"Classification Report of ML Algo")

  #Confusion Matrix
  cm = confusion_matrix(Y_test, Y_pred)
  print(cm)

  # performace parameters
  accuracy = accuracy_score(Y_test, Y_pred)
  precision = precision_score(Y_test, Y_pred)
  recall = recall_score(Y_test, Y_pred)
  #f1 = f1_score(Y_test, Y_pred)
  results.append(
      {
        "Model Name": name,
          "Accuracy": accuracy,
          "Precision": precision,
          "Recall": recall,
          #"F1": f1
      }
  )

  # Summary of the model
  results_df = pd.DataFrame(results)
  print("Summary of all the Models")
  print(results_df)

  # Visulaize the confusion matrix
  plt.figure(figsize=(5, 4))
  sns.heatmap(cm, annot=True)
  plt.xlabel("Predicted")
  plt.ylabel("Actual")
  plt.title("Confusion Matrix")
  plt.show()

  # Visual the result
  plt.figure(figsize=(12, 8))
  results_df.set_index("Model Name")[["Accuracy", "Precision", "Recall"]].plot(kind="bar", cmap="magma")
  plt.title("Visulization of Model Performance")
  plt.show()

import joblib

best_model_info = max(results, key=lambda x: x["Accuracy"])
best_model_name = best_model_info["Model Name"]
print(f"Best model is: {best_model_name}")

best_model = models[best_model_name]
best_model.fit(X_train, Y_train)

joblib.dump(best_model, 'titanic_model.pkl')
joblib.dump(encoder, 'sex_encoder.pkl')
print("Model and encoder saved successfully.")


