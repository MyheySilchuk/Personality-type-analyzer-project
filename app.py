import streamlit as st
import joblib
import numpy as np

# Load the pre-trained model
model = joblib.load("personality_type_analisation_model.pkl")

# Streamlit app title
st.title("🧠 Test: Introvert or Extrovert?")

# Streamlit app description
st.write("""
    This test will help you determine whether you are more of an introvert or an extrovert.
    Answer the questions honestly, and we'll give you your result!
""")
stage_fear = st.selectbox(
    "Do you feel fear or anxiety when speaking in front of an audience?",
    ["Yes", "No"]
)
stage_fear = 10 if stage_fear == "Yes" else 0
friends_size = st.number_input("How many friends do you have?")
drained = st.slider("How drained do you feel after social interaction?", 0, 10, 5)
post_freq = st.slider("How often do you post on social media?", 0, 10, 5)
social_events = st.slider("How often do you attend different events?", 0, 10, 5)
outside = st.slider("How often do you go outside per week?", 0, 10, 5)
alone_time = st.slider("How much time do you spend alone?", 0, 10, 5)

if st.button("Get Result"):
    # Form the input array
    features = np.array([[alone_time, stage_fear, social_events,
                          outside, drained, friends_size, post_freq]])

    # Prediction
    prediction = model.predict(features)[0]
    label = "Introvert 🧩" if prediction == 0 else "Extrovert 🎉"
    
    st.subheader(f"Your result: {label}")
    st.write("Thank you for taking the test! We hope it helps you understand yourself better.")
    st.write("Remember, this is just for fun and you shouldn't take the result too seriously!")