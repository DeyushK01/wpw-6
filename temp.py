import joblib
import os
import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from scipy.io import loadmat
from scipy import signal
from tqdm import tqdm
import neurokit2 as nk
from werkzeug.security import generate_password_hash, check_password_hash

# Load models
model = joblib.load('models/symptoms model/model.pkl')
model2 = load_model('models/wpw models/2d CNN/WPW_2d_att1.h5')

# Load dataset
def load_challenge_data(filename):
    x = loadmat(filename)
    data = np.asarray(x['val'], dtype=np.float64)
    new_file = filename.replace('.mat','.hea')
    input_header_file = os.path.join(new_file)
    with open(input_header_file,'r') as f:
        header_data = f.readlines()
    return data, header_data

def import_ecg_data(directory, ecg_len=5000, trunc="post", pad="post"):
    ecgs = []
    for ecgfilename in tqdm(sorted(os.listdir(directory))):
        filepath = directory + os.sep + ecgfilename
        if filepath.endswith(".mat"):
            data, header_data = load_challenge_data(filepath)
            data = pad_sequences(data, maxlen=ecg_len, truncating=trunc, padding=pad)
            ecgs.append(data)
    return np.asarray(ecgs)

def remove_some_ecgs(ecg_arr):
    delete_list = []
    for i in tqdm(range(len(ecg_arr))):
        if np.all(ecg_arr[i].T[0] == 1):
            delete_list.append(i)
    ecg_arr = np.delete(ecg_arr, delete_list, axis=0)
    return ecg_arr

def resample_beats(beats):
    rsmp_beats = []
    for i in beats:
        f = signal.resample(np.asarray(i), 250)
        rsmp_beats.append(f)
    return np.asarray(rsmp_beats)

def median_beat(beat_dict):
    beats = [i['Signal'] for i in beat_dict.values()]
    rsmp_beats = resample_beats(beats)
    return np.median(rsmp_beats, axis=0)

def process_ecgs(raw_ecg):    
    processed_ecgs = []
    for i in tqdm(range(len(raw_ecg))):
        leadII = raw_ecg[i][1]
        leadII_clean = nk.ecg_clean(leadII, sampling_rate=500, method="neurokit")
        r_peaks = nk.ecg_findpeaks(leadII_clean, sampling_rate=500, method="neurokit", show=False)
        twelve_leads = []
        for j in raw_ecg[i]:
            try:
                beats = nk.ecg_segment(j, rpeaks=r_peaks['ECG_R_Peaks'], sampling_rate=500, show=False)
                twelve_leads.append(median_beat(beats))
            except:
                twelve_leads.append(np.ones(250) * np.nan)
        processed_ecgs.append(twelve_leads)
    return np.asarray(processed_ecgs)

def remove_nans(ecg_arr):
    new_arr = []
    for i in tqdm(ecg_arr):
        twelve_lead = []
        for j in i:
            if np.isnan(j[0]):
                j = np.ones(250)
            twelve_lead.append(j)
        new_arr.append(twelve_lead)
    return np.asarray(new_arr)

# Streamlit application
st.title("ECG Detection and Symptom Checker")

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Detect ECG", "Check Symptoms", "Information"])

if page == "Home":
    st.write("Welcome to the ECG Detection and Symptom Checker app!")

elif page == "Detect ECG":
    st.header("ECG Detection")
    name = st.text_input("Enter Name")
    mat_file = st.file_uploader("Upload .mat File", type=["mat"])
    hea_file = st.file_uploader("Upload .hea File", type=["hea"])
    
    if st.button("Analyze ECG"):
        if name and mat_file and hea_file:
            os.makedirs(f'Uploads/{name}', exist_ok=True)
            mat_path = f'Uploads/{name}/{mat_file.name}'
            hea_path = f'Uploads/{name}/{hea_file.name}'
            with open(mat_path, "wb") as f:
                f.write(mat_file.getbuffer())
            with open(hea_path, "wb") as f:
                f.write(hea_file.getbuffer())

            x = import_ecg_data(f'Uploads/{name}')
            x = process_ecgs(x)
            x = remove_nans(x)
            x = remove_some_ecgs(x)
            x = np.moveaxis(x, 1, -1)
            pr = model2.predict(x)

            prediction = "Positive" if pr[0][1] > pr[0][0] else "Negative"
            st.success(f"Prediction: {prediction}")
        else:
            st.warning("Please provide name and upload both .mat and .hea files.")

elif page == "Check Symptoms":
    st.header("Symptom Checker")
    with st.form("symptom_form"):
        hr = st.number_input("Heart Rate (H.R)")
        cp = st.number_input("Chest Pain (C.P)")
        db = st.number_input("Difficulty Breathing (D.B)")
        dizziness = st.number_input("Dizziness")
        faint = st.number_input("Fainting")
        fatigue = st.number_input("Fatigue")
        anxiety = st.number_input("Anxiety")
        submitted = st.form_submit_button("Check")

    if submitted:
        features = np.array([[hr, cp, db, dizziness, faint, fatigue, anxiety]])
        feat_names = ['H.R', 'C.P', 'D.B', 'Dizziness', 'Faint', 'Fatigue', 'Anxiety']
        x = pd.DataFrame(features, columns=feat_names)
        pr = model.predict(x)

        result = "to get an ECG" if pr == 1 else "no need to worry"
        st.success(f"Patient has {result}")

elif page == "Information":
    st.header("Information")
    st.write("Provide useful information about the application here.")

if __name__ == "__main__":
    st.write("Streamlit app running")
