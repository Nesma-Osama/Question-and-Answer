# Configurable QA Chatbot

A simple and configurable **Question & Answer chatbot** built with **Python, Streamlit, and LangChain**.

The application allows users to choose an LLM provider, select a model, and configure model parameters before asking questions.

> 🚧 **Project Status: Under Development**
>
> This project is still being improved. More features such as **chat history, and other improvements** will be added in future versions.

---

## Features

- 💬 Question & Answer chatbot
- 🤖 Support for multiple LLM providers
- 🔧 Select the model from available models
- 🔑 User-provided API key
- 🌡️ Adjustable temperature
- 🔢 Adjustable maximum output tokens
- ⚡ Uses LangChain's `init_chat_model`
- 🖥️ Interactive Streamlit interface
- ❌ Basic error handling
- 🔌 Provider-independent model initialization
- 📄 PDF document loading
- ✂️ Document chunking using RecursiveCharacterTextSplitter
- 🔎 Semantic similarity search
- 🗄️ Chroma vector database
- 🧠 Hugging Face embeddings
- 🎯 Cross-encoder reranking
- 📚 RAG-based question answering
- 📋 Displays retrieved documents/context used to answer the question
- 📊 LangSmith integration for tracing and monitoring
---

## Technologies

- **Python** – Main programming language
- **Streamlit** – Web interface
- **LangChain** – LLM application framework
- **LangChain Classic** – Prompt templates
- **LangSmith** – LLM tracing, debugging, monitoring, and evaluation
- **Groq** – LLM provider
- **OpenAI** – LLM provider
- **python-dotenv** – Environment variable management
- **Chroma** – Vector database
- **Hugging Face** – Embedding and reranking models

---

## Project Structure

```text
Question-and-Answer/
│
├── app.py
├── chatbot.py
├── requirements.txt
├── .env
├── research_papers/
├── .gitignore
└── README.md
