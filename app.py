from flask import Flask, render_template, request
import pickle
import pandas as pd

with open("model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

app = Flask(__name__)
app.config['SECRET_KEY'] = "!2345@abc"

FEATURE_COLUMNS = [
    'Pregnancies',
    'Glucose',
    'BloodPressure',
    'SkinThickness',
    'Insulin',
    'BMI',
    'DiabetesPedigreeFunction',
    'Age'
]


@app.route("/")
def home():
    return render_template('index.html')


@app.route("/predict", methods=['GET', 'POST'])
def predict():
    if request.method != "POST":
        return render_template("result.html", no_value=1)

    try:
        values = {
            'Pregnancies': float(request.form.get('pregnancies')),
            'Glucose': float(request.form.get('glucose')),
            'BloodPressure': float(request.form.get('bloodpressure')),
            'SkinThickness': float(request.form.get('skinthickness')),
            'Insulin': float(request.form.get('insulin')),
            'BMI': float(request.form.get('bmi')),
            'DiabetesPedigreeFunction': float(request.form.get('dpf')),
            'Age': float(request.form.get('age'))
        }

        features = pd.DataFrame([values], columns=FEATURE_COLUMNS)
        prediction = model.predict(features)[0]

        prob = None
        if hasattr(model, 'predict_proba'):
            try:
                prob = round(float(model.predict_proba(features)[0][1] * 100), 1)
            except Exception:
                prob = None

        return render_template(
            "result.html",
            prediction=int(prediction),
            values=values,
            probability=prob
        )
    except (TypeError, ValueError) as exc:
        print(f"Prediction input error: {exc}")
        return render_template("result.html", no_value=1)


if __name__ == '__main__':
    app.run(debug=True)
