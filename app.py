import joblib
import flask
import os
from flask import render_template,request,redirect, url_for
from werkzeug.utils import secure_filename
import pandas as pd
import numpy as np
import keras
from distutils.log import debug
from fileinput import filename
from flask import *
from tensorflow.keras.preprocessing.sequence import pad_sequences
from scipy.io import loadmat
from scipy import signal
from tqdm import tqdm
import neurokit2 as nk

app =flask.Flask(__name__)
model=joblib.load('models/symptoms model/model.pkl','rb')
model2=keras.models.load_model('models/wpw models/2d CNN/WPW_2d_att1.h5')
@app.route('/')
@app.route('/#')
@app.route('/index')
def index():
    return render_template('index.html')
@app.route('/detect') 
def detect():
    return render_template('templates/Detect.html')
@app.route('/Detect',methods=['POST'])
def ECG():
    if request.method == 'POST':
        name=request.form['name']
        f = request.files['mat']
        file1=f.filename
        g= request.files['hea']
        file2=g.filename
        p=os.path.join('Uploads',name)
        print(p)
        os.mkdir(p)
        f.save(os.path.join(p, file1))
        g.save(os.path.join(p, file2))
        x= import_ecg_data(p)
        x = process_ecgs(x)
        x=remove_nans(x)
        x=remove_some_ecgs(x)
        x=np.moveaxis(x, 1, -1)
        print(x.shape)
        print(x)
        pr=model2.predict(x)
        print(pr[0][0])
        print(pr[0][1])
        if(pr[0][0]>pr[0][1]):
                prediction='Negative'
        elif(pr[0][0]<pr[0][1]):
                prediction='Positive'
        print(prediction)
        time.sleep(5)
        return render_template('detect.html',pred='{}'.format(prediction))
def remove_some_ecgs(ecg_arr):
    delete_list = []
    for i in tqdm(range(len(ecg_arr))):
        if np.all(ecg_arr[i].T[0]==1):
            delete_list.append(i)
    ecg_arr = np.delete(ecg_arr,delete_list,axis=0)
    return ecg_arr
def resample_beats(beats):
    rsmp_beats=[]
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
        #print(i['Signal'])
        beats.append(i['Signal'])
    beats = np.asarray(beats)
    rsmp_beats = resample_beats(beats)
    med_beat = np.median(rsmp_beats,axis=0)
    return med_beat

def process_ecgs(raw_ecg):    
    processed_ecgs=[]
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
                beats = np.ones(250)*np.nan
                twelve_leads.append(beats)
        #twelve_leads = np.asarray(twelve_leads)
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
    new_file = filename.replace('.mat','.hea')
    input_header_file = os.path.join(new_file)
    with open(input_header_file,'r') as f:
        header_data=f.readlines()
    return data, header_data
def import_ecg_data(directory, ecg_len = 5000, trunc="post", pad="post"):
    print("Starting ECG import..")
    ecgs = []
    for ecgfilename in tqdm(sorted(os.listdir(directory))):
        filepath = directory + os.sep + ecgfilename
        if filepath.endswith(".mat"):
            data, header_data = load_challenge_data(filepath)
            data = pad_sequences(data, maxlen=ecg_len, truncating=trunc,padding=pad)
            ecgs.append(data)
    print("Finished!")
    return np.asarray(ecgs)
@app.route('/sy')
def Symptoms():
    return render_template('sy.html')
@app.route('/ce')
def Centres():
    return render_template('ce.html')
@app.route('/Tr')
def Treat():
    return render_template('Tr.html')
@app.route('/sy',methods=['POST'])
def check():
    in_fea=[int(x) for x in request.form.values()]
    inp=[np.array(in_fea)]
    feat=['H.R','C.P','D.B','Dizziness','Faint','Fatigue','Anxiety']
    x=pd.DataFrame(inp,columns=feat)
    pr=model.predict(x)
    if(pr==1):
        p='to get an Ecg'
    else:
        p='no need to worry'
    return render_template('sy.html',pred='Patient has {}'.format(p))
if __name__=="__main__":
    app.run(debug=True)
