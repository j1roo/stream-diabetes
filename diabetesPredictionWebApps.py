import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import pickle
from sklearn.naive_bayes import GaussianNB

img = Image.open('diabetess.jpg')
img = img.resize((700, 418))
st.image(img, use_column_width=False)

st.sidebar.header('Input Data')

#Upload File CSV untuk parameter inputan
upload_file = st.sidebar.file_uploader('Upload your CSV file', type=['csv'])
if upload_file is not None:
    inputan = pd.read_csv(upload_file)
else:
    def input_user():
        gender = st.sidebar.selectbox('Gender', ('Male', 'Female'))
        age = st.sidebar.slider('Age', 0.08, 80.0, 40.0)
        hypertension = st.sidebar.slider('Hypertension', 0, 1, 0)
        heart_disease = st.sidebar.slider('Heart Disease', 0, 1, 0)
        smoking_history = st.sidebar.selectbox('Smoking History', ('never', 'former', 'current', 'No Info', 'not current'))
        bmi = st.sidebar.slider('BMI', 10.0, 95.7, 40.0)
        HbA1c_level = st.sidebar.slider('HbA1c', 3.5, 9.0, 5.0)
        blood_glucose_level = st.sidebar.slider('Blood Glucose Level', 80, 300, 150)
        data = {
            'gender' : gender,
            'age' : age,
            'hypertension' : hypertension,
            'heart_disease' : heart_disease,
            'smoking_history' : smoking_history,
            'bmi' : bmi,
            'HbA1c_level' : HbA1c_level,
            'blood_glucose_level' : blood_glucose_level
        }
        fitur = pd.DataFrame(data, index=[0])
        return fitur
    inputan = input_user()

#menggabungkan inputan dan dataset diabetes prediction
diabetesPrediction_raw = pd.read_csv('diabet.csv')
diabetesPredictions = diabetesPrediction_raw.drop(columns=['diabetes'])
df = pd.concat([inputan, diabetesPredictions], axis=0)

#encode untuk fitur ordinal
encode = ['gender', 'smoking_history']
for col in encode :
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df, dummy], axis = 1)
    del df[col]
df = df[:1] #ambil baris pertama (input data user)

#menampilkan parameter hasil inputan
st.subheader('Input Parameters')

if upload_file is not None:
    st.write("""
    True = 1\n
    False = 0
    """)
    st.write(df)
else:
    st.write('Menunggu file csv diunggah. saat ini menggunakan sampel input')
    st.write("""
    True = 1\n
    False = 0
    """)
    st.write(df)

#load model NBC
load_model = pickle.load(open('modNBC_diabetes_prediction.pkl', 'rb'))

#terapkan NBC
prediksi = load_model.predict(df)
prediksi_proba = load_model.predict_proba(df)

st.subheader('Deskripsi Kelas Label')
st.write("""
Positive = 1\n
Negative = 0
""")
status_diabetes = np.array([0,1])
st.write(status_diabetes)

st.subheader('Hasil Prediksi (Diabetes Prediction)')
st.write("""
Positive = 1\n
Negative = 0
""")
st.write(status_diabetes[prediksi])

st.subheader('Kemungkinan Hasil yang Diprediksi (Diabetes Prediction)')
st.write("""
Positive = 1\n
Negative = 0
""")
st.write(prediksi_proba)

