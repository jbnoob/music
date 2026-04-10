import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import cv2 
import numpy as np 
import mediapipe as mp 
from keras.models import load_model
import webbrowser

# Load the saved emotion model and labels from the data training step
model  = load_model("model.h5")
label = np.load("labels.npy")

# Initialize Mediapipe holistic model for face and hand landmarks
holistic = mp.solutions.holistic
hands = mp.solutions.hands
holis = holistic.Holistic()
drawing = mp.solutions.drawing_utils

# Manage the session state to control camera behavior
if "run" not in st.session_state:
    st.session_state["run"] = "true"

# Try to load the previously saved prediction, if it exists
try:
    emotion = np.load("emotion.npy")
except:
    emotion = ""

# Determine whether the camera should be running or stopped
if not emotion:
    st.session_state["run"] = "true"
else:
    st.session_state["run"] = "false"

# Create a class to process each frame coming from the webcam
class emotion_processor:
    def recv(self, frame):
        # Convert incoming frame to a BGR numpy array
        frm = frame.to_ndarray(format="bgr24")
        frm = cv2.flip(frm, 1)

        # Detect holistic landmarks
        res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))

        lst = []

        # Iterate through face and hand landmarks and store them in lst
        if res.face_landmarks:
            for i in res.face_landmarks.landmark:
                lst.append(i.x - res.face_landmarks.landmark[3].x)
                lst.append(i.y - res.face_landmarks.landmark[3].y)

        if res.left_hand_landmarks:
            for i in res.left_hand_landmarks.landmark:
                lst.append(i.x - res.left_hand_landmarks.landmark[4].x)
                lst.append(i.y - res.left_hand_landmarks.landmark[4].y)
        else:
            for i in range(42):
                lst.append(0.0)

        if res.right_hand_landmarks:
            for i in res.right_hand_landmarks.landmark:
                lst.append(i.x - res.right_hand_landmarks.landmark[4].x)
                lst.append(i.y - res.right_hand_landmarks.landmark[4].y)
        else:
            for i in range(42):
                lst.append(0.0)

        # Convert list to array and make a prediction
        lst = np.array(lst).reshape(1, -1)
        pred = label[np.argmax(model.predict(lst))]
        print(pred)
        
        # Display the emotion on the video frame
        cv2.putText(frm, pred, (50, 50), cv2.FONT_ITALIC, 1, (255, 0, 0), 2)
        
        # Save the predicted emotion locally to survive Streamlit refreshes
        np.save("emotion.npy", np.array([pred]))

        # Return the processed frame
        return av.VideoFrame.from_ndarray(frm, format="bgr24")

# UI Elements
language = st.text_input("Language")
singer = st.text_input("singer")

# Open the webcam conditionally if inputs are filled and run state is true
if language and singer and st.session_state["run"] != "false":
    webrtc_streamer(key="key", desired_playing_state=True, video_processor_factory=emotion_processor)

btn = st.button("Recommend me song")

# Handle the button click event
if btn:
    if not emotion:
        # Show a warning if an emotion hasn't been detected yet
        st.warning("Please let me capture your emotion first")
    else:
        # Open YouTube with a formatted search query 
        webbrowser.open(f"https://www.youtube.com/results?search_query={language}+{emotion}+song+{singer}")
        
        # Reset the emotion to an empty string so the process can be repeated
        np.save("emotion.npy", np.array([""]))
        st.session_state["run"] = "false"