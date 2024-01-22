import streamlit as st
import openai
from sentence_transformers import util
import pandas as pd
import numpy as np
import torch
import os
import time
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt

openai.api_key = st.secrets["OPENAI_API_KEY"]
  
st.markdown("# MATH WORLD")

#MODEL_NAME = "all-MiniLM-L12-v2"
EMBEDDING_CSV = "embeddings.csv"
QUESTION_COLUMN_NAME = "Question"
ANSWER_COLUMN_NAME = "Answer"
FEEDBACK_CSV = "feedback.csv"
FEEDBACK_COLUMN_ONE = "Date"
FEEDBACK_COLUMN_TWO = "Question"
FEEDBACK_COLUMN_THREE = "Feedback"
FEEDBACK_COLUMN_FOUR = "Answer Provided"
NUM_CHUNKS = 5
MODEL = "text-embedding-ada-002"

STYLING = """
    <style>
        .about {
            text-align: justify;
            box-shadow: 5px 10px 18px grey;
            padding: 10px;
            border-radius: 10px;
            margin-bottom: 20px;
        }

        .question {
            text-align: justify;
            box-shadow: 5px 10px 18px grey;
            padding: 10px;
            border-radius: 10px;
            margin-top: 20px;
            margin-bottom: 40px;
        }

        .answers {
            text-align: justify;
            box-shadow: 5px 10px 18px grey;
            padding: 10px;
            border-radius: 10px;
            margin-top: 20px;
            margin-bottom: 40px;
        }

        .question:hover {
            background-color: #92C9E2;
        }

        .about:hover {
            background-color: #92E2D2;
        }

        .answers:hover {
            background-color: #E8EF82;
        }
    </style>
"""
#set css styles
st.markdown(STYLING, unsafe_allow_html=True)

if 'user' not in st.session_state:
    st.session_state.user = None

if 'question' not in st.session_state:
    st.session_state.question = None

#disable in session state
if 'disabled' not in st.session_state:
    st.session_state.disabled = False

def disabled():
    st.session_state.disabled = True

#this is to download the transformer model
#this takes some times
# this is cached unless otherwise model will download again and again
# function to extract embeddings
def extract_embedding(text):
    print("Helllo")
    get_embedding = openai.embeddings.create(model= MODEL,input=text)
    return get_embedding.data[0].embedding


@st.cache_data
def load_embeddings():
    embeddings = pd.concat(list(map(pd.read_csv, ["OpenAIchunk_{}.csv".format(chunk_number) for chunk_number in range(NUM_CHUNKS)])), ignore_index = True)
    return np.array(embeddings.iloc[:,2:].values), embeddings[QUESTION_COLUMN_NAME].to_list(), embeddings[ANSWER_COLUMN_NAME].to_list()


#load models
embeddings, questions_db, answers_db = load_embeddings()


def get_similar_question(query, num_questions, question_embeddings, main_questions, main_answers):
    answers_dict = []
    questions_dict = []
    #embed the query
    query_embedding = extract_embedding(query)
    query_embedding = np.array(query_embedding).astype(np.float32)
    question_embeddings = question_embeddings.astype(np.float32)
    #get the similarity
    cos_score = util.cos_sim(query_embedding, question_embeddings)
    top_scores = torch.topk(cos_score,max(1, num_questions))

    #get the index array
    indexes = top_scores[1][0]
    #get related question
    for question_index in indexes:
        questions_dict.append(main_questions[question_index])
        answers_dict.append(main_answers[question_index])

    return questions_dict, answers_dict


def answers_holders(*args, answers):

    for index, (question, answer) in enumerate(zip(args, answers)):
        question_div = f"""
            <div class="answers">
                <h5>Similar Question: {index+1}</h5>
                <p><strong>{question}</strong></p>
            </div>
        """
 #set the div
        st.markdown(question_div, unsafe_allow_html = True)
        time.sleep(2)
        answer = answer.replace('.', '**').replace('####', '**')
        steps = answer.split('**')
        for step in steps:
            if step.strip():
                answer_div = f"""
                <div class="step">
                <p><strong>{step.strip()}</strong></p>
                </div>
                """
                st.markdown(answer_div, unsafe_allow_html = True)
                time.sleep(2)
        time.sleep(2)

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

#app design
st.title("Math Minds")

#set an images
st.image("https://d2zhlgis9acwvp.cloudfront.net/images/uploaded/mathematicians.jpg", caption = "Mathematics")

#set description
examples_div = """
    <div class="Examples">
        <h3>Example questions to help you get started!</h3>
        <ul>
            <li>A box has 5 dozen bottles of apple juice and half a dozen more bottle of orange juice than apple juice. How many bottles are in the box?</li>
            <li>Tim can read 10 pages of a book in 30 minutes.  How many hours will it take him to read 120 pages?</li>
            <li>What is 30 more than a quarter of 40?</li>
            <li>The price of a home is $120 per square foot. The house is 3000 square feet and the shed is 1000 square feet. How much is this property?</li>
            <li>If 40 out of 100 people like basketball and out of those that like it, 30% play it, how many people would you expect play basketball out of a group of 300 people?</li>
        </ul>
    </div>
"""
#set the container
st.markdown(examples_div, unsafe_allow_html = True)

#create tabs
tab1, tab2 = st.tabs(["Questions 🧮", "Dashboard 📊"])


#tab1
with tab1:
    #set title
    st.header("Let's Discover the Question Bank")
    st.subheader("Please Enter the Question ❓")
    #get the question
    question = st.text_input('Please Enter Your Question', placeholder = "Your Question")

    if question:
        question_div = f"""
            <div class="question">
                <h4>Your Question</h4>
                <p><strong>{question}<strong></p>
            </div>
        """

        #set the question
        st.markdown(question_div, unsafe_allow_html = True)

        #set the slides
        st.subheader("Choose the Number of Questions")
        num_questions = st.slider("Number of Questions You want", min_value = 1, max_value = 5, value = 1)

        if num_questions:
            #get the similar questions
            similar_questions, similar_answers = get_similar_question([question], num_questions, embeddings, questions_db, answers_db)

            #set the answers
            answers_holders(*similar_questions, answers =similar_answers)

            #set the user to true
            st.session_state.user = True
            st.session_state.question = question

            #define feedback state
            feedback_state = False
            button_state = True

            with st.sidebar:
                st.title("Feedback")
                st.write("Your feedback is much appreciated for improvemment of the web application.")
                st.subheader("How helpful was this AI Generated Response towards helping you solve this probelm?")

                #create columns
                col1, col2, col3, col4, col5, col6, col7 = st.columns(7)

                with col1:
                    #set feedback buttons
                    if st.button("0 (Actively Harmful)", type = "primary", on_click=disabled, disabled=st.session_state.disabled, use_container_width = True):
                        feedback_state = update_feedback(datetime.now(), question, "0")

                with col2:
                    if st.button("1 (Very Harmful)", type = "primary", on_click=disabled, disabled=st.session_state.disabled, use_container_width = True):
                        feedback_state = update_feedback(datetime.now(), question, "1")

                with col3:
                    if st.button("2 (Somewhat Harmful)", type = "primary", on_click=disabled, disabled=st.session_state.disabled, use_container_width = True):
                        feedback_state = update_feedback(datetime.now(), question, "2")
                      
                with col4:
                    if st.button("3 (Unlikely to help,but unlikely to hurt)", type = "primary", on_click=disabled, disabled=st.session_state.disabled, use_container_width = True):
                        feedback_state = update_feedback(datetime.now(), question, "3")     
                      
                with col5:
                    if st.button("4 (Somewhat Helpful)", type = "primary", on_click=disabled, disabled=st.session_state.disabled, use_container_width = True):
                        feedback_state = update_feedback(datetime.now(), question, "4")
                      
                with col6:
                    if st.button("5 (Very Helpful)", type = "primary", on_click=disabled, disabled=st.session_state.disabled, use_container_width = True):
                        feedback_state = update_feedback(datetime.now(), question, "5")
                      
                with col7:
                    if st.button("6 (Extremely Helpful)", type = "primary", on_click=disabled, disabled=st.session_state.disabled, use_container_width = True):
                        feedback_state = update_feedback(datetime.now(), question, "6")
 
                #say thank you if the feedback is given
                if feedback_state:
                    st.header("Thank You For Your Feedback!")
with tab2:
    st.header("Feedback Analysis")
    st.subheader("Feedback Data")

    if not os.path.exists(FEEDBACK_CSV):
        st.error("Take Some Feedbacks from Users to view the Analytics 🚨")
    else:
        #read the data
        feedbacks = pd.read_csv(FEEDBACK_CSV)
        st.dataframe(feedbacks)
        #count plot
        sns.set_theme(style="whitegrid", palette="pastel")
        chart_fig = plt.figure(figsize=(5,5))
        sns.countplot(data = feedbacks, x = FEEDBACK_COLUMN_THREE)
        plt.xlabel("Feedback")
        plt.ylabel("Count")
        plt.title("Feedback Distribution")
        st.pyplot(chart_fig)
