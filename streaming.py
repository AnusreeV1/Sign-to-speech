import streamlit as st
import av
from gtts import gTTS
from io import BytesIO
import threading
import cv2
import numpy as np
import time
from string import ascii_uppercase
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from streamlit_webrtc import (
    ClientSettings,
    VideoTransformerBase,
    WebRtcMode,
    webrtc_streamer,
)
#from textblob import TextBlob
#from spellchecker import SpellChecker

#spell = SpellChecker()

# Prepare data generator for standardizing frames before sending them into the model.
st.set_page_config(layout="wide")
@st.cache(allow_output_mutation=True)
def update_slider():
    return {"slide":0}

data_generator = ImageDataGenerator(samplewise_center=True, samplewise_std_normalization=True)

# Loading the model.
MODEL_NAME = 'models/asl_alphabet_{}.h5'.format(9575)
model = load_model(MODEL_NAME)

# Setting up the input image size and frame crop size.
IMAGE_SIZE = 200
CROP_SIZE = 400

# Creating list of available classes stored in classes.txt.
classes_file = open("classes.txt")
classes_string = classes_file.readline()
classes = classes_string.split()
classes.sort()

def main():
    st.header("Sign to text!")
    
class DrawBounds(VideoTransformerBase):
    x_:int
    y_:int
    def __init__(self) -> None:
        
        self.frame_lock = threading.Lock()
    def transform(self,frame:av.VideoFrame) -> np.ndarray:
        img=frame.to_ndarray(format="bgr24")
        cv2.rectangle(img, (self.x_, 0), (CROP_SIZE+self.x_, CROP_SIZE), (0, 255, 0), 3)
        return img


class VideoTransformer(VideoTransformerBase):
    frame_lock: threading.Lock
    x_:int
    def __init__(self) -> None:
        self.classes_dict={}
        # DrawBounds().__init__()
        for i in classes:
            self.classes_dict[i]=0
        self.blank_flag=0
        self.word=" "
        self.current_symbol=""
        self.sentence=""
        self.predicted_class=""
        
    def predict(self):
        # global word,current_symbol,sentence,classes_dict,classes
        
        global classes
        self.current_symbol=self.predicted_class
        self.classes_dict[self.current_symbol]+=1
        #print(current_symbol)
        if(self.current_symbol=='nothing' and self.classes_dict[self.current_symbol]>50):
            #print("Nothing")
            #word=spell.correction(word)
            if(len(self.sentence)==0):
                return
            mp3_fp=BytesIO()
            tts = gTTS(self.word)
            tts.write_to_fp(mp3_fp)
            for i in classes:
                self.classes_dict[i]=0
            if(len(self.sentence)>0):
                self.sentence+=" "
            
            self.sentence+=self.word
            self.word=""
            return
        if(self.classes_dict[self.current_symbol]>50):
            for i in classes:
                if(i==self.current_symbol):
                    continue
                elif(abs(self.classes_dict[self.current_symbol]-self.classes_dict[i])<20):
                    for i in classes:
                        self.classes_dict[i]=0
                        return
        
            if(self.current_symbol=='nothing'):
                for i in classes:
                    self.classes_dict[i]=0
                if(len(self.sentence)>0):
                    self.sentence+=" "
                self.sentence+=self.word
                self.word=""
            else:
                for i in classes:
                    self.classes_dict[i]=0
                self.word+=self.current_symbol
                return
    def transform(self, frame:av.VideoFrame) -> np.ndarray:
        global model

    # Target area where the hand gestures should be.
        img=frame.to_ndarray(format="bgr24")
        cv2.rectangle(img, (self.x_, 0), (CROP_SIZE+self.x_, CROP_SIZE), (0, 255, 0), 3)
        #cv2.rectangle(img, (100, 5), (100, 400), (0, 200, 0), 3)
        
        
        # Preprocessing the frame before input to the model.
        cropped_image = img[0:CROP_SIZE, 0:CROP_SIZE]
        resized_frame = cv2.resize(cropped_image, (IMAGE_SIZE, IMAGE_SIZE))
        reshaped_frame = (np.array(resized_frame)).reshape((1, IMAGE_SIZE, IMAGE_SIZE, 3))
        frame_for_model = data_generator.standardize(np.float64(reshaped_frame))

        # Predicting the frame.
        prediction = np.array(model.predict(frame_for_model))
        self.predicted_class = classes[prediction.argmax()]     # Selecting the max confidence index.
        self.predict()
        cv2.putText(img,self.word,(10, 450), 1, 2, (255, 255, 0), 2, cv2.LINE_AA)
        return img


draw_bounds=False
draw_bounds=st.checkbox("The box is in correct position", value=False)
common_words=['Good morning','Good afternoon','Good night','Hello','Hi','How are you?','I am fine','What are you doing?','Thank you','Sorry','Okay']
slider_value=update_slider()
x_axis=slider_value["slide"]
if not draw_bounds:
    webrtc_ctx1=webrtc_streamer(key="draw_box",mode=WebRtcMode.SENDRECV,video_transformer_factory=DrawBounds,async_transform=True,)
    
    x_axis = st.slider("X-axis", 1, 200,slider_value["slide"],2)
    slider_value["slide"]=x_axis
    if (webrtc_ctx1.video_transformer) :
        webrtc_ctx1.video_transformer.x_=x_axis
    
    
if draw_bounds:
    
    col1, col2 = st.beta_columns(2)
    with col2:
        st.markdown("Here are the commmon phrases, Just click on the button and play the audio\n")
        for i in common_words:
            if(st.button(i,key=i)):
                mp3_fp = BytesIO()
                tts = gTTS(i)
                tts.write_to_fp(mp3_fp)   
                st.audio(mp3_fp)
    
    with col1:
        speak=st.button("Speak")
        st_audio=st.empty()
        webrtc_ctx = webrtc_streamer(
            key="Hemashirisha123",
            mode=WebRtcMode.SENDRECV,
            video_transformer_factory=VideoTransformer,
            async_transform=True,
        )
     


        if webrtc_ctx.video_transformer:
            #slider_value=update_slider()
            webrtc_ctx.video_transformer.x_=slider_value["slide"]
            

  

