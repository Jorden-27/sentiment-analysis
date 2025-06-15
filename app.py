from flask import Flask, render_template, request
import pickle
import re

# Flask setup
app = Flask(__name__)

# Load models and vectorizer
logreg = pickle.load(open("logreg_model.pkl", "rb"))
nb = pickle.load(open("nb_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
accuracies = pickle.load(open("accuracies.pkl", "rb"))

# Preprocess function
def preprocess(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text.strip()

# Prediction logic
def predict_sentiment(model, text):
    text = preprocess(text)
    vector = vectorizer.transform([text])
    prediction = model.predict(vector)[0]
    return "Positive" if prediction == 1 else "Negative"

@app.route("/", methods=["GET", "POST"])
def index():
    result = ""
    accuracy = None
    if request.method == "POST":
        tweet = request.form["tweet"]
        model_choice = request.form["model"]
        
        if model_choice == "logreg":
            result = predict_sentiment(logreg, tweet)
            accuracy = accuracies["logreg"]
        else:
            result = predict_sentiment(nb, tweet)
            accuracy = accuracies["nb"]
        
        return render_template("index.html", tweet=tweet, result=result,
                               accuracy=round(accuracy * 100, 2), model=model_choice)
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
