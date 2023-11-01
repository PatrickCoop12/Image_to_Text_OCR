import streamlit as st
import pandas as pd

st.set_page_config(page_title="PostGame Extract", page_icon=":tada:", layout="wide")

#Header

st.subheader("This is a site demo for POC")
st.title("Our Project's aim is to make alleviate the hassle of manual transcription and hard to read, handwritten documents in the sports world")
st.write("[Learn More>](https://google.com)")

file_upload = st.sidebar.file_uploader('Please Upload a File')


if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    response = f"Echo: {prompt}"
# Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
# Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})


st.text_input("question", key="question")

# You can access the value at any point with:
st.session_state.question


