import flask
import os
import math
from google.cloud import storage
from google.cloud import speech
from google.cloud.speech import enums
from google.cloud.speech import types
from pydub import AudioSegment
import pandas as pd
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from matplotlib.pyplot import specgram
import keras
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.callbacks import ModelCheckpoint
app = flask.Flask(__name__)
def upload_to_gcloud(bucket_name, source_file_name, destination_blob_name):
    try:
        storage_client = storage.Client()
    except Exception as e:
        print(e)
    
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)    
def format_transcript(results, audio_file):
    def format_time(seconds, offset=0): #time conversion/formatting for timestamps
        frac, whole = math.modf(seconds)
        f = frac * 1000
        m, s = divmod(whole, 60)
        h, m = divmod(m, 60)
        return "%d:%02d:%02d,%03d" % (h, m, s, (f + offset * 1000))
    def chunks(l, n):
        for i in range(0, len(l), n):
            yield l[i:i + n]
    file = open( audio_file + ".srt", "w")
    counter = 0 # Used for numbering lines in file
    for result in results:
        print(result)
        alternatives = result.alternatives
        for alternative in alternatives:
            print(alternative)
            words = alternative.words
            print(words)
            if len(words) < 14:
                transcript = alternative.transcript
                start_time = words[0].start_time
                end_time = words[-1].end_time
                start_time_seconds = start_time.seconds + start_time.nanos * 1e-9
                end_time_seconds = end_time.seconds + end_time.nanos * 1e-9
                counter += 1
                file.write(str(counter) + '\n')
                file.write(format_time(start_time_seconds) + ' --> ' + format_time(end_time_seconds) + '\n')
                file.write(transcript + "\n\n")
            else:
                chunk = list(chunks(words, 14))
                for words in chunk:
                    start_time = words[0].start_time
                    end_time = words[-1].end_time
                    start_time_seconds = start_time.seconds + start_time.nanos * 1e-9
                    end_time_seconds = end_time.seconds + end_time.nanos * 1e-9
                    section = ''
                    for word_info in words:
                        section += word_info.word + " "
                    counter += 1
                    file.write(str(counter) + '\n')
                    file.write(format_time(start_time_seconds) + ' --> ' + format_time(end_time_seconds) + '\n')
                    file.write(section + "\n\n")
    file.close()
def transcribe_gcs(ogg_file,ext):
    def strip(text, suffix): # Strip .mp4 suffix from file name
        if not text.endswith(suffix):
            return text
        # else
        return text[:len(text) - len(suffix)]

    stripped_name = strip(ogg_file, "."+ext)
    audio_file_path = stripped_name #Create audio file
    if audio_file_path:
        bucket_name = 'mostafa-lebod' # Your gcloud bucket name
        audio_file_name = audio_file_path[1:] + "." + ext
        upload_to_gcloud(bucket_name, source_file_name=audio_file_path + "."+ext, destination_blob_name=audio_file_name)
        CLIENT = speech.SpeechClient()
        audio = types.RecognitionAudio(uri="gs://" + bucket_name +'/'+ audio_file_name )
        ENCODING = [enums.RecognitionConfig.AudioEncoding.LINEAR16, enums.RecognitionConfig.AudioEncoding.FLAC,enums.RecognitionConfig.AudioEncoding.MULAW,enums.RecognitionConfig.AudioEncoding.AMR,enums.RecognitionConfig.AudioEncoding.AMR_WB,enums.RecognitionConfig.AudioEncoding.OGG_OPUS,enums.RecognitionConfig.AudioEncoding.SPEEX_WITH_HEADER_BYTE]
        SAMPLE_RATE_HERTZ = [8000, 12000, 16000, 24000,41000, 48000]
        results=[]
        temp = ''
        for enco in ENCODING:
            for rate in SAMPLE_RATE_HERTZ:
                config = types.RecognitionConfig(
                    encoding=enco,
                    sample_rate_hertz=rate,
                    language_code='en-US')

                response = []
                try:
                    response = CLIENT.long_running_recognize(config, audio)
                    result = response.result()
                    results = result.results
                except Exception as e:
                    print(e)
                if(len(results) > 0):
                      if temp == results[0].alternatives[0].transcript:
                        return results[0].alternatives[0].transcript
                      else:
                        temp = results[0].alternatives[0].transcript
        if temp != '':
            return temp
        return 'ERROR: Couldn\'t identify any voice'
    else:
        return
def soundEmo(path):
    try:
        df = pd.DataFrame(columns=['feature'])
        bookmark=0
        try:
            X, sample_rate = librosa.load(path,duration=2.5,sr=22050*2,offset=0.2)
        except Exception as e:
            print(e)
        sample_rate = np.array(sample_rate)
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13), axis=0)
        feature = mfccs
        df.loc[bookmark] = [feature]
        bookmark=bookmark+1
        df_new = pd.DataFrame(df['feature'].values.tolist())
        df_new=df_new.fillna(0)
        X_test = np.array(df_new)
        x_testcnn= np.expand_dims(X_test, axis=2)
        model = keras.models.load_model(os.path.join(app.root_path, 'Emotion_Voice_Detection_Model.h5'))
        preds = model.predict(x_testcnn, batch_size=1, verbose=1)
        preds1=preds.argmax(axis=1)
        return preds1[0]
    except Exception as e:
        print(e)

@app.route("/soundtotext", methods=["POST"])
def soundtotext():
    data = {"success": False}
    if flask.request.method == "POST":
        if flask.request.files.get("audio"):
            try:
                os.environ['GOOGLE_APPLICATION_CREDENTIALS'] =os.path.join(app.root_path, 'gc-creds.json') 
                file = flask.request.files['audio']
                file.save(os.path.join(app.root_path, file.filename))
                transcript = transcribe_gcs(os.path.join(app.root_path, file.filename),file.filename.split('.')[1])
                data["transcript"] = transcript
                data["success"] = True
            except Exception as e:
                print(e)
    return flask.jsonify(data)

@app.route("/soundtoemo", methods=["POST"])
def soundtoemo():
    data = {"success": False}
    if flask.request.method == "POST":
        if flask.request.files.get("audio"):
            try:
                file = flask.request.files['audio']
                file.save(os.path.join(app.root_path, file.filename))
                emo = soundEmo(os.path.join(app.root_path, file.filename))
                data['emo'] = str(emo)
                data["success"] = True
            except Exception as e:
                print(e)
    return flask.jsonify(data)
@app.route('/')
def index():
    return "<h1>Welcome to our server !!</h1>"

if __name__ == '__main__':
    app.run(Threaded=True)
