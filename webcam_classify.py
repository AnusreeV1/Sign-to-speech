import cv2
import numpy as np
import time
from string import ascii_uppercase
import time
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from textblob import TextBlob
from spellchecker import SpellChecker

spell = SpellChecker()

# Prepare data generator for standardizing frames before sending them into the model.
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
classes.sort()  # The predict function sends out output in sorted order.

# Preparing cv2 for webcam feed
cap = cv2.VideoCapture(0)
i=0
classes_dict={}
for i in classes:
    classes_dict[i]=0
blank_flag=0
word=""
current_symbol=""
sentence=""
def predict():
    global word,current_symbol,sentence,classes_dict,classes
    current_symbol=predicted_class
    classes_dict[current_symbol]+=1
    #print(current_symbol)
    if(current_symbol=='nothing' and classes_dict[current_symbol]>50):
        #print("Nothing")
        for i in classes:
            classes_dict[i]=0
        if(len(sentence)>0):
            sentence+=" "
        word=spell.correction(word)
        sentence+=spell.correction(word)
        word=""
        return
    if(classes_dict[current_symbol]>100):
        for i in classes:
            if(i==current_symbol):
                continue
            elif(abs(classes_dict[current_symbol]-classes_dict[i])<20):
                for i in classes:
                    classes_dict[i]=0
                    return
       
        if(current_symbol=='nothing'):
            for i in classes:
                classes_dict[i]=0
            if(len(sentence)>0):
                sentence+=" "
                sentence+=word
                word=""
        else:
            for i in classes:
                classes_dict[i]=0
            word+=current_symbol
            return
                
    
    
        
while(True):
    # Capture frame-by-frame.
    #time.sleep(5)
    ret, frame = cap.read()

    # Target area where the hand gestures should be.
    cv2.rectangle(frame, (0, 0), (CROP_SIZE, CROP_SIZE), (0, 255, 0), 3)
    
    # Preprocessing the frame before input to the model.
    cropped_image = frame[0:CROP_SIZE, 0:CROP_SIZE]
    resized_frame = cv2.resize(cropped_image, (IMAGE_SIZE, IMAGE_SIZE))
    reshaped_frame = (np.array(resized_frame)).reshape((1, IMAGE_SIZE, IMAGE_SIZE, 3))
    frame_for_model = data_generator.standardize(np.float64(reshaped_frame))

    # Predicting the frame.
    prediction = np.array(model.predict(frame_for_model))
    predicted_class = classes[prediction.argmax()]     # Selecting the max confidence index.
    predict()
   
    #print(sentence,end="")
    cv2.putText(frame,word,(10, 450), 1, 2, (255, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow('frame', frame)
    k = cv2.waitKey(1) & 0xFF
    if k == ord('q'):
        break
    #time.sleep(1)

prediction_probability = prediction[0, prediction.argmax()]
textBlb = TextBlob(sentence)            # Making our first textblob
textCorrected = textBlb.correct()   # Correcting the text
print(textCorrected)
cap.release()
cv2.destroyAllWindows()
