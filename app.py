import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import base64

# -------------------------------
# Function to set background image
# -------------------------------

def set_bg(image_file):
    with open(image_file, "rb") as img:
        encoded = base64.b64encode(img.read()).decode()

    bg = f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{encoded}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }}

    .user {{
        background-color:#3b82f6;
        padding:10px;
        border-radius:10px;
        margin:5px;
        color:white;
        width:fit-content;
    }}

    .bot {{
        background-color:#22c55e;
        padding:10px;
        border-radius:10px;
        margin:5px;
        color:white;
        width:fit-content;
    }}
    </style>
    """
    st.markdown(bg, unsafe_allow_html=True)


# -------------------------------
# Set Background
# -------------------------------

set_bg("back.avif")

# -------------------------------
# Page Settings
# -------------------------------

st.set_page_config(page_title="AI College Chatbot", layout="wide")

st.title("🎓 AI College Helpdesk Chatbot")

# -------------------------------
# Load Dataset
# -------------------------------

data = pd.read_csv("dataset.csv")

questions = data["question"]
answers = data["answer"]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(questions)

# -------------------------------
# Chatbot Logic
# -------------------------------

def chatbot_response(user_input):

    user_vec = vectorizer.transform([user_input])
    similarity = cosine_similarity(user_vec, X)

    index = similarity.argmax()

    return answers[index]


# -------------------------------
# Session Memory
# -------------------------------

if "chat" not in st.session_state:
    st.session_state.chat = []

if "input_text" not in st.session_state:
    st.session_state.input_text = ""


# -------------------------------
# Sidebar Quick Questions
# -------------------------------

st.sidebar.title("📚 Quick Questions")

quick_questions = [
"What are college timings?",
"Where is the library?",
"How to apply for admission?",
"What courses are offered?",
"Is hostel available?"
]

for q in quick_questions:
    if st.sidebar.button(q):
        st.session_state.input_text = q


if st.sidebar.button("🧹 Clear Chat"):
    st.session_state.chat = []


# -------------------------------
# Chat Interface
# -------------------------------

st.subheader("Chat")

for role, message in st.session_state.chat:

    if role == "user":
        st.markdown(f'<div class="user">👤 {message}</div>', unsafe_allow_html=True)

    else:
        st.markdown(f'<div class="bot">🤖 {message}</div>', unsafe_allow_html=True)


# -------------------------------
# Input Box
# -------------------------------

user_input = st.text_input(
    "Ask your question",
    value=st.session_state.input_text
)

# -------------------------------
# Send Button
# -------------------------------

if st.button("Send"):

    if user_input:

        response = chatbot_response(user_input)

        st.session_state.chat.append(("user", user_input))
        st.session_state.chat.append(("bot", response))

        st.session_state.input_text = ""

        st.rerun()