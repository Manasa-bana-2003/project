from flask import Flask, render_template, request
import nltk
import pickle
from nltk.sentiment.vader import SentimentIntensityAnalyzer #
# Load the model and tokenizer
with open("news-Predict.pkl",'rb') as f:
    rf_count = pickle.load(f)

with open("rf_model_word.pkl",'rb') as f:
    rf_word = pickle.load(f)

app = Flask(__name__)
#model = load_model('nextword1.h5')
nltk.download('vader_lexicon')
@app.route('/')
def me():
    return render_template('index.html')

@app.route('/predict', methods=['GET','POST'])
def predict():
    text = request.form['review']
    score = SentimentIntensityAnalyzer().polarity_scores(text)
    if score["compound"] >= 0.25:
      predict= "Comment is Positive..!"
    elif -0.25 <= score["compound"] < 0.25:
      predict= "Comment is Neutral..! "
    else:
      predict = "Comment is Negative..!"
                   
    return render_template('index.html',predict=predict)

if __name__=="__main__":
    app.run(use_reloader=True,debug=True)

