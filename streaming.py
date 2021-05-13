import queue
import threading
import time
import urllib.request
from collections import deque
from io import BytesIO
from pathlib import Path
from string import ascii_uppercase
from typing import List


import av
import cv2
import numpy as np
import pydub
import streamlit as st
from bokeh.models import CustomJS
from bokeh.models.widgets import Button
from gtts import gTTS
from streamlit_bokeh_events import streamlit_bokeh_events
from streamlit_webrtc import (AudioProcessorBase,
                              ClientSettings,
                              VideoTransformerBase, WebRtcMode,
                              webrtc_streamer)
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

HERE = Path(__file__).parent
st.set_page_config(layout="wide")
WEBRTC_CLIENT_SETTINGS = ClientSettings(
    media_stream_constraints={"video": True, "audio": False},
)

def download_file(url, download_to: Path, expected_size=None):
    # Don't download the file twice.
    # (If possible, verify the download using the file length.)
    if download_to.exists():
        if expected_size:
            if download_to.stat().st_size == expected_size:
                return
        else:
            st.info(f"{url} is already downloaded.")
            if not st.button("Download again?"):
                return

    download_to.parent.mkdir(parents=True, exist_ok=True)

    # These are handles to two visual elements to animate.
    weights_warning, progress_bar = None, None
    try:
        weights_warning = st.warning("Downloading %s..." % url)
        progress_bar = st.progress(0)
        with open(download_to, "wb") as output_file:
            with urllib.request.urlopen(url) as response:
                length = int(response.info()["Content-Length"])
                counter = 0.0
                MEGABYTES = 2.0 ** 20.0
                while True:
                    data = response.read(8192)
                    if not data:
                        break
                    counter += len(data)
                    output_file.write(data)

                    # We perform animation by overwriting the elements.
                    weights_warning.warning(
                        "Downloading %s... (%6.2f/%6.2f MB)"
                        % (url, counter / MEGABYTES, length / MEGABYTES)
                    )
                    progress_bar.progress(min(counter / length, 1.0))
    # Finally, we remove these visual elements by calling .empty().
    finally:
        if weights_warning is not None:
            weights_warning.empty()
        if progress_bar is not None:
            progress_bar.empty()

MODEL_URL = "https://github.com/mozilla/DeepSpeech/releases/download/v0.9.3/deepspeech-0.9.3-models.pbmm"  # noqa
LANG_MODEL_URL = "https://github.com/mozilla/DeepSpeech/releases/download/v0.9.3/deepspeech-0.9.3-models.scorer"  # noqa
MODEL_LOCAL_PATH = HERE / "models/deepspeech-0.9.3-models.pbmm"
LANG_MODEL_LOCAL_PATH = HERE / "models/deepspeech-0.9.3-models.scorer"
download_file(MODEL_URL, MODEL_LOCAL_PATH, expected_size=188915987)
download_file(LANG_MODEL_URL, LANG_MODEL_LOCAL_PATH, expected_size=953363776)
lm_alpha = 0.931289039105002
lm_beta = 1.1834137581510284
beam = 100
def app_sst(model_path: str, lm_path: str, lm_alpha: float, lm_beta: float, beam: int):
    webrtc_ctx = webrtc_streamer(
        key="speech-to-text",
        mode=WebRtcMode.SENDONLY,
        audio_receiver_size=1024,
        client_settings=ClientSettings(
            rtc_configuration={
                "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
            },
            media_stream_constraints={"video": False, "audio": True},
        ),
    )

    status_indicator = st.empty()

    if not webrtc_ctx.state.playing:
        return

    status_indicator.write("Loading...")
    text_output = st.empty()
    stream = None

    while True:
        if webrtc_ctx.audio_receiver:
            if stream is None:
                from deepspeech import Model

                model = Model(model_path)
                model.enableExternalScorer(lm_path)
                model.setScorerAlphaBeta(lm_alpha, lm_beta)
                model.setBeamWidth(beam)

                stream = model.createStream()

                status_indicator.write("Model loaded.")

            sound_chunk = pydub.AudioSegment.empty()
            try:
                audio_frames = webrtc_ctx.audio_receiver.get_frames(timeout=1)
            except queue.Empty:
                time.sleep(0.1)
                status_indicator.write("No frame arrived.")
                continue

            status_indicator.write("Running. Say something!")

            for audio_frame in audio_frames:
                sound = pydub.AudioSegment(
                    data=audio_frame.to_ndarray().tobytes(),
                    sample_width=audio_frame.format.bytes,
                    frame_rate=audio_frame.sample_rate,
                    channels=len(audio_frame.layout.channels),
                )
                sound_chunk += sound

            if len(sound_chunk) > 0:
                sound_chunk = sound_chunk.set_channels(1).set_frame_rate(
                    model.sampleRate()
                )
                buffer = np.array(sound_chunk.get_array_of_samples())
                stream.feedAudioContent(buffer)
                text = stream.intermediateDecode()
                text_output.markdown(f"**Text:** {text}")
        else:
            status_indicator.write("AudioReciver is not set. Abort.")
            break

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
                #st.markdown(self.sentence)
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
    
    col1, col2, col3 = st.beta_columns(3)
    with col3:
        st.header("Here are the commmon phrases, Just click on the button and play the audio\n")
        for i in common_words:
            if(st.button(i,key=i)):
                mp3_fp = BytesIO()
                tts = gTTS(i)
                tts.write_to_fp(mp3_fp)   
                st.audio(mp3_fp)

    with col2:
        # tts_button = Button(label="Speak", width=100)

        # tts_button.js_on_event("button_click", CustomJS(code=f"""
        #     var u = new SpeechSynthesisUtterance();
        #     u.text = "{text}";
        #     u.lang = 'en-US';

        #     speechSynthesis.speak(u);
        #     """))
        # st.bokeh_chart(tts_button)

        # st.header("Speech to text")
        # stt_button = Button(label="Speak", width=100)
        # stt_button.js_on_event("button_click", CustomJS(code="""
        #     var recognition = new webkitSpeechRecognition();
        #     recognition.continuous = true;
        #     recognition.interimResults = true;
        
        #     recognition.onresult = function (e) {
        #         var value = "";
        #         for (var i = e.resultIndex; i < e.results.length; ++i) {
        #             if (e.results[i].isFinal) {
        #                 value += e.results[i][0].transcript;
        #             }
        #         }
        #         if ( value != "") {
        #             document.dispatchEvent(new CustomEvent("GET_TEXT", {detail: value}));
        #         }
        #     }
        #     recognition.start();
        #     """))

        # result = streamlit_bokeh_events(
        #     stt_button,
        #     events="GET_TEXT",
        #     key="listen",
        #     refresh_on_update=False,
        #     override_height=75,
        #     debounce_time=0)

        # if result:
        #     if "GET_TEXT" in result:
        #         st.write(result.get("GET_TEXT"))
        st.header("Speech to text")
        app_sst(
            str(MODEL_LOCAL_PATH), str(LANG_MODEL_LOCAL_PATH), lm_alpha, lm_beta, beam
        )


    with col1:
        st.header("Sign to speech")
        st_audio=st.empty()
        webrtc_ctx = webrtc_streamer(
            key="Hemashirisha123",
            mode=WebRtcMode.SENDRECV,
            video_transformer_factory=VideoTransformer,
            async_transform=True,
            client_settings=WEBRTC_CLIENT_SETTINGS,
        )
     
        if webrtc_ctx.video_transformer:
            #slider_value=update_slider()
            webrtc_ctx.video_transformer.x_=slider_value["slide"]
            if st.button('Speak'):
                webrtc_ctx.video_transformer.sentence += ""
                sen = webrtc_ctx.video_transformer.sentence
                st.markdown(sen)
                mp4 = BytesIO()
                tts = gTTS(sen)
                tts.write_to_fp(mp4)
                st.audio(mp4)
