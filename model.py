import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import joblib

# 1. Memuat dataset dari file Excel
data = pd.read_excel('data_diabetes_puskesmas_tegalbuleud.xlsx')

# 2. Preprocessing data
# Memisahkan fitur dan target
X = data.drop('Hasil Diagnosa (0: Non-diabetes, 1: Diabetes)', axis=1)
y = data['Hasil Diagnosa (0: Non-diabetes, 1: Diabetes)']

# Menormalkan fitur-fitur numerik
scaler = StandardScaler()
X[['Kadar Glukosa Darah', 'Tekanan Darah', 'Indeks Massa Tubuh (BMI)']] = scaler.fit_transform(
    X[['Kadar Glukosa Darah', 'Tekanan Darah', 'Indeks Massa Tubuh (BMI)']])

# 3. Membagi dataset menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Inisialisasi dan melatih model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 5. Evaluasi model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Akurasi Model: {accuracy}')

# 6. Menyimpan model yang sudah dilatih
joblib.dump(model, 'diabetes_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
