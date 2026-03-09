import streamlit as st
from ingestion.data_loader import load_data
from advisor.advisor_engine import generate_advice, prepare_vector_store, vector_store

st.set_page_config(page_title="💰 AI Finance Advisor", layout="wide")

st.title("💰 AI Business Finance Advisor")

# ----------------------------
# LOAD DATA WITH CACHE
# ----------------------------
@st.cache_data
def get_data():
    return load_data()

data = get_data()

# ----------------------------
# INITIALIZE VECTOR STORE
# ----------------------------
vector_store = prepare_vector_store(data)

# ----------------------------
# CHAT SESSION STATE
# ----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# display chat history
for role, message in st.session_state.messages:
    with st.chat_message(role):
        st.write(message)

# user input
user_input = st.chat_input("Ask a financial question...")

if user_input:
    # display user message
    st.session_state.messages.append(("user", user_input))
    with st.chat_message("user"):
        st.write(user_input)

    # spinner while AI is generating advice
    with st.spinner("AI is analyzing your business data..."):
        response = generate_advice(user_input, data, vector_store)

    # save and display AI response
    st.session_state.messages.append(("assistant", response))
    with st.chat_message("assistant"):
        st.write(response)