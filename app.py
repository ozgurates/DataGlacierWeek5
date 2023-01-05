from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

p = open("model.pickle", "rb")
regressor = pickle.load(p)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    studytime = request.form.get("studytime")
    failures = request.form.get("failures")
    abs = request.form.get("absence")
    G1 = request.form.get("G1")
    G2 = request.form.get("G2")
    features = [studytime, failures, abs, G1, G2]
    test_df = pd.DataFrame([features], columns=[
                           'studytime', 'failures', 'absence', 'G1', 'G2'])

    pred = regressor.predict(test_df)

    return render_template('index.html', pred=pred)


if __name__ == '__main__':
    app.run(debug=True)
