# Imported Libraries
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import warnings
from flask import Flask, request
import cv2
import numpy as np
import re
from keras.models import load_model
from PIL import Image
from keras.layers import Input, LSTM, Dense
import random

warnings.filterwarnings('ignore')
le = LabelEncoder()
oneH = OneHotEncoder()

import Chatbot as CB
import Anxiety as anxiety
import Depression as depression
import Stress as stress
import General_Self_Efficacy_Scale as GSE

app = Flask(__name__)

@app.route('/chatbots', methods=['POST'])
def chatbots():
    try:
        sentence = request.form.get("chat")
        print(sentence)
        return {'data' : CB.start_chat(sentence)}
    except Exception as e:
        return {'data': 'An Error Occurred during fetching Api : '+ str(e)}, 400

@app.route('/chatbot', methods=['POST'])
def chatbot():
    try:
        return {'data' : "Hello"}
    except Exception as e:
        return {'data': 'An Error Occurred during fetching Api : '+ str(e)}, 400

@app.route('/GSEQ', methods=['POST'])
def GSEQ():
    GSE_Answers_text = request.form.get("GSE")
    GSE_Answers = GSE_Answers_text.split(", ")
    Predicted_GSE = GSE.predict_Positive_negative(GSE_Answers)
    return Predicted_GSE

@app.route('/PDSQ', methods=['POST'])
def PDSQ():
    PDS_Answers_txt = request.form.get("PDS")
    PDS_Answers = PDS_Answers_txt.split(", ")
    Depression_prediction = depression.predict_depression_status(PDS_Answers)
    return Depression_prediction

@app.route('/stressQ', methods=['POST'])
def stressQ():
    stress_Answers_txt = request.form.get("stress")
    stress_Answers = stress_Answers_txt.split(", ")
    Stress_prediction = stress.stress_depression_status(stress_Answers)
    return Stress_prediction

@app.route('/anxietyQ', methods=['POST'])
def anxietyQ():
    anxiety_Answers_txt = request.form.get("anxiety")
    anxiety_Answers = anxiety_Answers_txt.split(", ")
    Anxiety_prediction = anxiety.predict_Anxiety_status(anxiety_Answers)
    return Anxiety_prediction


@app.route('/emotionImage', methods=['POST'])
def emotionImage():
        try:
            image = request.files['file']
            img = Image.open(image)  # load with Pillow
            img.save('new.jpeg')
            ff = facefeature()
            val = ff.predictImage('new.jpeg')
            return {'data' : val}
        except Exception as e:
            return {'data': 'An Error Occurred during fetching Api : '+ str(e)},

@app.route('/emotionTest', methods=['POST'])
def emotionTest():
        try:
            num1 = random.randint(0, 6)
            ff = facefeature()
            return {'data' : ff.get_labels()[num1]}
        except Exception as e:
            return {'data': 'An Error Occurred during fetching Api : '+ str(e)}, 400

class facefeature():
    def predictImage(self, y):
        try:
          Filepath = "mysite/Data/"
          cascPath= Filepath+"abc.xml"
          emotion_model = "mysite/Data/Emotion.hdf5"
          model = load_model(emotion_model, compile=compile)
          PADDING = 40
          faceCascade = cv2.CascadeClassifier(cascPath)
          emotion_labels = self.get_labels()
          img = cv2.imread("new.jpeg")
          gray_image_array = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
          gray_img = self.pad(gray_image_array)

          emotions = []
          faces = faceCascade.detectMultiScale(
                  gray_image_array,
                  scaleFactor=1.1,
                  minNeighbors=5,
                  minSize=(30, 30))
          if len(faces) == 1:
                    gray_img = self.pad(gray_image_array)
                    for face_coordinates in faces:
                        face_coordinates = self.tosquare(face_coordinates)
                        x1, x2, y1, y2 = self.apply_offsets(face_coordinates)

                        # adjust for padding
                        x1 += PADDING
                        x2 += PADDING
                        y1 += PADDING
                        y2 += PADDING
                        x1 = np.clip(x1, a_min=0, a_max=None)
                        y1 = np.clip(y1, a_min=0, a_max=None)

                    gray_face = gray_img[max(0, y1 - PADDING):y2 + PADDING,
                                        max(0, x1 - PADDING):x2 + PADDING]
                    gray_face = gray_img[y1:y2, x1:x2]

                    model.make_predict_function()

                    try:
                      gray_face = cv2.resize(gray_face, model.input_shape[1:3])
                    except Exception as e:
                      print("Cannot resize "+str(e))
                    gray_face = np.expand_dims(np.expand_dims(gray_face, 0), -1)
                    emotion_prediction = model.predict(gray_face)[0]
                    labelled_emotions = {
                        emotion_labels[idx]: round(float(score), 2)
                        for idx, score in enumerate(emotion_prediction)
                        }

                    emotions.append(
                        dict(box=face_coordinates, emotions=labelled_emotions)
                        )
                    top_emotions  = [max(e["emotions"], key=lambda key: e["emotions"][key]) for e in emotions]
                    print(top_emotions)
                    return top_emotions[0]
        except Exception as e:
            return 'An Error Occurred during getting prediction : '+ str(e)


    def transform(self, X, y):
      return "fit"

    def get_labels(self):
      return {
                0: "angry",
                1: "disgust",
                2: "fear",
                3: "happy",
                4: "sad",
                5: "surprise",
                6: "neutral",
            }

    def tosquare(self, bbox):
            """Convert bounding box to square by elongating shorter side."""
            x, y, w, h = bbox
            if h > w:
                diff = h - w
                x -= diff // 2
                w += diff
            elif w > h:
                diff = w - h
                y -= diff // 2
                h += diff
            if w != h:
                print(f"{w} is not {h}")

            return (x, y, w, h)

    def apply_offsets(self, face_coordinates):
      x, y, width, height = face_coordinates
      x_off, y_off = (10, 10)
      return (x - x_off, x + width + x_off, y - y_off, y + height + y_off)

    def preprocess_input(self, x, v2=False):
            x = x.astype("float32")
            x = x / 255.0
            if v2:
                x = x - 0.5
                x = x * 2.0
            return x

    def pad(self, image):
            PADDING = 40
            row, col = image.shape[:2]
            bottom = image[row - 2 : row, 0:col]
            mean = cv2.mean(bottom)[0]

            padded_image = cv2.copyMakeBorder(
                image,
                top = PADDING,
                bottom = PADDING,
                left = PADDING,
                right= PADDING,
                borderType=cv2.BORDER_CONSTANT,
                value=[mean, mean, mean],
            )
            return padded_image
