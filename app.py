import streamlit as st
import pandas as pd
import joblib

# loading the trained model
model = joblib.load("personality_rf_model.pkl")

# app title
st.set_page_config(
    page_title="Personality Predictor",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="auto",
)

st.title("Personality Type Predictor")
st.subheader("Enter your social and behavioral traits: ")

# input form
time_spent_alone = st.slider("time spent alone daily in hours", 0, 11, 5)
stage_fear = st.selectbox("do you have stage fear?", ["yes", "maybe", "no"])
social_event_attendance = st.slider("social event attendance (0-10)", 0, 10, 5)
going_outside = st.slider("frequency of going outside (0-7)", 0, 7, 3)
drained_after_socializing = st.selectbox(
    "do you feel drained after socializing", ["yes", "sometimes", "no"]
)
friend_circle_size = st.slider("number of friends", 0, 15, 5)
post_frequency = st.slider("social media post frequency (0-10)", 0, 10, 5)

# convert inputs to match model format
if stage_fear == "yes":
    stage_fear_bins = 1
else:
    stage_fear_bins = 0

if drained_after_socializing == "yes":
    drained_bin = 1
else:
    drained_bin = 0

# creating a dataframe for the model
input_data = pd.DataFrame(
    {
        "Time_spent_Alone": [time_spent_alone],
        "Stage_fear": [stage_fear_bins],
        "Social_event_attendance": [social_event_attendance],
        "Going_outside": [going_outside],
        "Drained_after_socializing": [drained_bin],
        "Friends_circle_size": [friend_circle_size],
        "Post_frequency": [post_frequency],
    }
)

if st.button("Predict Personality"):
    prediction = model.predict(input_data)[0]
    if prediction == 1:
        result = "Extrovert"
        description = (
            "You're energized by social interaction and enjoy being around people."
        )
    else:
        result = "Introvert"
        description = (
            "You tend to recharge in solitude and prefer fewer social interactions."
        )
    st.success(f"Predicted personality: {result}")
    st.info(description)
