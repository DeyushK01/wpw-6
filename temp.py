import joblib
import os
import pandas as pd
import numpy as np
from keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from scipy.io import loadmat
from scipy import signal
from tqdm import tqdm
import neurokit2 as nk
import time
import streamlit as st
import matplotlib.pyplot as plt

# Load models
model = joblib.load('models/symptoms model/model.pkl')
model2 = load_model('models/wpw models/2d CNN/WPW_2d_att1.h5')

# Helper functions
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
        i = np.asarray(i)
        i = i[~np.isnan(i)]
        f = signal.resample(i, 250)
        rsmp_beats.append(f)
    rsmp_beats = np.asarray(rsmp_beats)
    return rsmp_beats

def median_beat(beat_dict):
    beats = []
    for i in beat_dict.values():
        beats.append(i['Signal'])
    beats = np.asarray(beats)
    rsmp_beats = resample_beats(beats)
    med_beat = np.median(rsmp_beats, axis=0)
    return med_beat

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
                med_beat = median_beat(beats)
                twelve_leads.append(med_beat)
            except:
                beats = np.ones(250) * np.nan
                twelve_leads.append(beats)
        processed_ecgs.append(twelve_leads)
    processed_ecgs = np.asarray(processed_ecgs)
    return processed_ecgs

def remove_nans(ecg_arr):
    new_arr = []
    for i in tqdm(ecg_arr):
        twelve_lead = []
        for j in i:
            if j[0] != j[0]:
                j = np.ones(250)
            twelve_lead.append(j)
        new_arr.append(twelve_lead)
    new_arr = np.asarray(new_arr)
    return new_arr

def load_challenge_data(filename):
    x = loadmat(filename)
    data = np.asarray(x['val'], dtype=np.float64)
    return data

def import_ecg_data(directory, ecg_len=5000, trunc="post", pad="post"):
    st.write("Starting ECG import..")
    ecgs = []
    for ecgfilename in tqdm(sorted(os.listdir(directory))):
        filepath = directory + os.sep + ecgfilename
        if filepath.endswith(".mat"):
            data = load_challenge_data(filepath)
            data = pad_sequences(data, maxlen=ecg_len, truncating=trunc, padding=pad)
            ecgs.append(data)
    st.write("Finished!")
    return np.asarray(ecgs)

# Streamlit app
def main():
    st.title("ECG Analysis and Symptom Prediction")
    
    # ECG Analysis Section
    st.header("ECG Analysis")
    uploaded_file = st.file_uploader("Upload ECG .mat file", type=["mat"])
    
    if uploaded_file is not None:
        path = 'Uploads'
        os.makedirs(path, exist_ok=True)
        file_path = os.path.join(path, secure_filename(uploaded_file.name))
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        x = import_ecg_data(path)
        x = process_ecgs(x)
        x = remove_nans(x)
        x = remove_some_ecgs(x)
        x = np.moveaxis(x, 1, -1)
        pr = model2.predict(x)
        
        if pr[0][0] > pr[0][1]:
            prediction = 'Negative'
        else:
            prediction = 'Positive'
        
        st.write(f"Prediction: {prediction}")
    
    # Symptom Prediction Section
    st.header("Symptom Prediction")
    with st.form("symptom_form"):
        feat = ['H.R', 'C.P', 'D.B', 'Dizziness', 'Faint', 'Fatigue', 'An']
        inputs = {f: st.text_input(f) for f in feat}
        submit_button = st.form_submit_button("Submit")
        
        if submit_button:
            inp = [np.array([inputs[f] for f in feat])]
            x = pd.DataFrame(inp, columns=feat)
            pr = model.predict(x)
            st.write(f"Prediction: {pr}")

if __name__ == "__main__":
    main()
