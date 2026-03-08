import streamlit as st
from ingestion.data_loader import load_data
from advisor.advisor_engine import generate_advice

st.set_page_config(page_title="AI Finance Advisor", layout="wide")

st.title("💰 AI Business Finance Advisor")

data = load_data()

if "messages" not in st.session_state:
    st.session_state.messages = []

# show chat history
for role, message in st.session_state.messages:
    with st.chat_message(role):
        st.write(message)

user_input = st.chat_input("Ask a financial question...")

if user_input:

    # show user message
    st.session_state.messages.append(("user", user_input))
    with st.chat_message("user"):
        st.write(user_input)

    # spinner while AI is generating
    with st.spinner("AI is analyzing your business data..."):

        response = generate_advice(user_input, data)

    # save response
    st.session_state.messages.append(("assistant", response))

    with st.chat_message("assistant"):
        st.write(response)