from flask import Flask, request, render_template
import pickle

# Load the vectorizer and the model from disk
with open("vectorizer.pkl", 'rb') as vec_file:
    vectorizer = pickle.load(vec_file)
    
with open("finalized_model.pkl", 'rb') as model_file:
    model = pickle.load(model_file)

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/prediction", methods=['GET', 'POST'])
def prediction():
    if request.method == "POST":
        news = request.form['news']
        predict = model.predict(vectorizer.transform([news]))
        return render_template("prediction.html", prediction_text="News headline is -> {}".format(predict[0]))
    else:
        return render_template("prediction.html")

if __name__ == "__main__":
    app.run(debug=True)
