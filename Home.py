import streamlit as st
import openai
from sentence_transformers import util
import pandas as pd
import numpy as np
import torch

openai.api_key = st.secrets["OPENAI_API_KEY"]
  
st.markdown("# MATH WORLD")

#MODEL_NAME = "all-MiniLM-L12-v2"
EMBEDDING_CSV = "embeddings.csv"
QUESTION_COLUMN_NAME = "Question"
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
    return np.array(embeddings.iloc[:,2:].values), embeddings[QUESTION_COLUMN_NAME].to_list()


#load models
embeddings, questions_db = load_embeddings()


def get_similar_question(query, num_questions, question_embeddings, main_questions):
    answers_dict = []
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
        answers_dict.append(main_questions[question_index])

    return answers_dict


def answers_holders(*args):

    for index, questions in enumerate(args):
        answers_div = f"""
            <div class="answers">
                <h5>Similar Question: {index+1}</h5>
                <p><strong>{questions}</strong></p>
            </div>
        """

        #set the div
        st.markdown(answers_div, unsafe_allow_html = True)


#app design
st.title("Math Minds")

#set an images
st.image("https://d2zhlgis9acwvp.cloudfront.net/images/uploaded/mathematicians.jpg", caption = "Mathematics")

#set description
about_div = """
    <div class="about">
        <h3>About the App</h3>
        <ul>
            <li>This web app is to sharpen your mathematical skills.</li>
            <li>You can copy paste any mathematical question and get similar question/questions.</li>
            <li>Can use them to enhance your skills.</li>
        </ul>
    </div>
"""

#set the container
st.markdown(about_div, unsafe_allow_html = True)

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
            similar_questions = get_similar_question([question], num_questions, embeddings, questions_db)

            #set the answers
            answers_holders(*similar_questions)

            #set the user to true
            st.session_state.user = True
            st.session_state.question = question

            #define feedback state
            feedback_state = False
            button_state = True
