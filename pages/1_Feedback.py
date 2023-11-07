import streamlit as st
import os 
import pandas as pd
from datetime import datetime


FEEDBACK_CSV = "feedback.csv"
FEEDBACK_COLUMN_ONE = "Date"
FEEDBACK_COLUMN_TWO = "Question"
FEEDBACK_COLUMN_THREE = "Feedback"

def update_feedback(interact_date, user_question, user_feedback):
    if not os.path.exists(FEEDBACK_CSV):
        feedback_data = pd.DataFrame(columns = [FEEDBACK_COLUMN_ONE, FEEDBACK_COLUMN_TWO, FEEDBACK_COLUMN_THREE])
        feedback_data.to_csv(FEEDBACK_CSV, index = False)

    feedback_df = pd.read_csv(FEEDBACK_CSV)

    metadata = {}
    metadata[FEEDBACK_COLUMN_ONE] = interact_date
    metadata[FEEDBACK_COLUMN_TWO] = user_question
    metadata[FEEDBACK_COLUMN_THREE] = user_feedback
    feedback_df_new = pd.concat([feedback_df, pd.DataFrame(metadata, index = [0])], ignore_index = True)
    feedback_df_new.to_csv(FEEDBACK_CSV, index = False)
    print("Saved successfully")
    return True


#disable in session state
if 'disabled' not in st.session_state:
    st.session_state.disabled = False

if 'question' not in st.session_state:
    st.session_state.question = None

if 'user' not in st.session_state:
    st.session_state.user = None

def disabled():
    st.session_state.disabled = True

# set the title
st.title("Feedback Page")

if st.session_state.question and st.session_state.user:
    # subheader
    st.subheader("Please Provide Your Feddbacks")
    
    feedback_state = None
    #create columns
    col1, col2 = st.columns(2)

    with col1:
        #set feedback buttons
        if st.button("Yes", type = "primary", on_click=disabled, disabled=st.session_state.disabled, use_container_width = True):
            feedback_state = update_feedback(datetime.now(), st.session_state.question, "Yes")

    with col2:
        if st.button("No", type = "primary", on_click=disabled, disabled=st.session_state.disabled, use_container_width = True):
            feedback_state = update_feedback(datetime.now(), st.session_state.question, "No")

    #say thank you if the feedback is given
    if feedback_state:
        st.header("Thank You For Your Feedback!")
else:
    st.error("Please Interact with the Web App First", icon = "🚨")
