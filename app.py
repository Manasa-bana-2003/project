from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import pickle
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import cross_val_score, GridSearchCV, cross_validate, train_test_split
# Load the model and tokenizer
df = pd.read_csv("reviews.csv", sep=",")

with open("news-Predict.pkl",'rb') as f:
    rf_count = pickle.load(f)

with open("rf_model_word.pkl",'rb') as f:
    rf_word = pickle.load(f)
def create_label(dataframe, dependent_var, independent_var):
  sia = SentimentIntensityAnalyzer()
  dataframe[independent_var] = dataframe[dependent_var].apply(lambda x: "pos" if sia.polarity_scores(x)["compound"] > 0 else "neg")
  dataframe[independent_var] = LabelEncoder().fit_transform(dataframe[independent_var])

  X = dataframe[dependent_var]
  y = dataframe[independent_var]

  return X, y
X, y = create_label(df, "Description", "sentiment_label")

def split_dataset(dataframe, X, y):
  train_x, test_x, train_y, test_y = train_test_split(X, y, random_state=1)
  return train_x, test_x, train_y, test_y

def create_features_count(train_x, test_x):
  # Count Vectors
  vectorizer = CountVectorizer()
  x_train_count_vectorizer = vectorizer.fit_transform(train_x)
  x_test_count_vectorizer = vectorizer.fit_transform(test_x)

  return x_train_count_vectorizer, x_test_count_vectorizer

train_x, test_x, train_y, test_y = split_dataset(df, X, y)

def create_features_TFIDF_word(train_x, test_x):
  # TF-IDF word
  tf_idf_word_vectorizer = TfidfVectorizer()
  x_train_tf_idf_word = tf_idf_word_vectorizer.fit_transform(train_x)
  x_test_tf_idf_word = tf_idf_word_vectorizer.fit_transform(test_x)

  return x_train_tf_idf_word, x_test_tf_idf_word

x_train_tf_idf_word, x_test_tf_idf_word = create_features_TFIDF_word(train_x, test_x)

app = Flask(__name__)
#model = load_model('nextword1.h5')

@app.route('/')
def me():
    return render_template('index.html')

@app.route('/predict', methods=['GET','POST'])
def predict():
    text = request.form['review']
    text = pd.Series(text)
    text = CountVectorizer().fit(train_x).transform(text)
    result = rf_count.predict(text)
    #result = rf_word.predict(text)
    if result==1:
        predict= "Comment is Positive"
        
    else: 
        predict = "Comment is Negative"
                    
    return render_template('index.html',predict=predict)

if __name__=="__main__":
    app.run(use_reloader=True,debug=True)
