import os
import streamlit as st
from rag_utils import get_rag_chain

# ✅ Set your Google API Key (via secrets or environment variable)
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

st.set_page_config(page_title="📄 Paper Q&A Chatbot", page_icon="🤖")
st.title("📄 Research Paper Chatbot with Gemini + FAISS")

st.markdown("Ask questions about the research paper stored in the app.")

if "rag_chain" not in st.session_state:
    with st.spinner("Loading model and vector store..."):
        st.session_state.rag_chain = get_rag_chain()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# User input
user_input = st.text_input("Your question:", placeholder="E.g., Summarize the key findings...")

if user_input:
    with st.spinner("Getting answer..."):
        response = st.session_state.rag_chain.invoke({"query": user_input})
        st.session_state.chat_history.append(("You", user_input))
        st.session_state.chat_history.append(("AI", response['result']))

# Display chat history
for role, message in st.session_state.chat_history:
    if role == "You":
        st.markdown(f"**🧑‍💻 {role}:** {message}")
    else:
        st.markdown(f"**🤖 {role}:** {message}")
