from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Memuat model dan scaler
model = joblib.load('diabetes_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/form')
def form():
    return render_template('form.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Mengambil data input dari form
        usia = int(request.form['usia'])
        pregnancies = int(request.form['pregnancies'])
        glucose = int(request.form['glucose'])
        blood_pressure = int(request.form['blood_pressure'])
        skin_thickness = int(request.form['skin_thickness'])
        insulin = int(request.form['insulin'])
        bmi = float(request.form['bmi'])
        diabetes_pedigree = float(request.form['diabetes_pedigree'])
        family_history = int(request.form['family_history'])

        # Membuat array untuk prediksi
        input_data = np.array([[usia, pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, family_history]])

        # Normalisasi data input
        input_data[:, 2:5] = scaler.transform(input_data[:, 2:5])

        # Prediksi menggunakan model
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

        return render_template('result.html', prediction=prediction, probability=probability)

if __name__ == '__main__':
    app.run(debug=True)
