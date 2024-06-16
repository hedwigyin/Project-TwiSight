import streamlit as st

from langchain.memory import ConversationBufferMemory
from utils import qa_agent

st.title("📑 TwiSight-Financial Document Insight Engine")

with st.sidebar:
    openai_api_key = st.text_input("Please enter your OpenAI API key：", type="password")
    st.markdown("[Get OpenAI API key](https://platform.openai.com/account/api-keys)")

if "memory" not in st.session_state:
    st.session_state["memory"] = ConversationBufferMemory(
        return_messages=True,
        memory_key="chat_history",
        output_key="answer"
    )

uploaded_file = st.file_uploader("Upload your PDF here：", type="pdf")
question = st.text_input("Ask your questions", disabled=not uploaded_file)

if uploaded_file and question and not openai_api_key:
    st.info("Please enter your OpenAI API key")

if uploaded_file and question and openai_api_key:
    with st.spinner("Your AI assistant is functioning，please wait..."):
        response = qa_agent(openai_api_key, st.session_state["memory"],
                            uploaded_file, question)
    st.write("## Answer")
    st.write(response["answer"])
    st.write("### Source")
    st.write(response["source_documents"])
    st.session_state["chat_history"] = response["chat_history"]

if "chat_history" in st.session_state:
    with st.expander("History chat"):
        for i in range(0, len(st.session_state["chat_history"]), 2):
            human_message = st.session_state["chat_history"][i]
            ai_message = st.session_state["chat_history"][i + 1]
            st.write(human_message.content)
            st.write(ai_message.content)
            if i < len(st.session_state["chat_history"]) - 2:
                st.divider()

