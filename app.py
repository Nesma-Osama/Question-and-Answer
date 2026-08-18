import streamlit as st
from chatbot import chatbot_func

st.title("Configurable QA Chatbot")
st.write("Go a head and ask any question but first write valid api key and settings ")
user_input = st.text_input("You")
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your API KEY", type="password")
model_provider = st.sidebar.selectbox("Model Provider", ["groq", "openai"], index=0)
groq_models = [
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
]
openai_models = [
    "gpt-5.4-mini",
    "gpt-5.4",
    "gpt-5.4-pro",
]
if model_provider == "groq":
    model_name = st.sidebar.selectbox("Model Name", groq_models, index=0)
else:
    model_name = st.sidebar.selectbox("Model Name", openai_models, index=0)
temperature = st.sidebar.slider(
    "Teamperature", min_value=0.0, max_value=1.0, step=0.1, value=0.5
)
max_tokens = st.sidebar.slider(
    "Max Tokens", min_value=50, max_value=500, step=10, value=250
)

if api_key:
    if user_input:
        response = chatbot_func(
            model_provider, api_key, model_name, temperature, max_tokens, user_input
        )
        st.write(response["answer"])
        with st.expander("Similarity Search"):
            for doc in response["context"]:
                st.write(doc.page_content)
                st.write("------------------------------")        
    else:
        st.write("Ask Question")
else:
    st.write("Configure Settings first")
