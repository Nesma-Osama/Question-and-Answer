from dotenv import load_dotenv
from langchain_classic.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_classic.retrievers.contextual_compression import (
    ContextualCompressionRetriever,
)
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
import streamlit as st

load_dotenv()


def get_model(model_provider, api_key, model_name, temperature, max_tokens):
    try:
        model = init_chat_model(
            model=model_name,
            model_provider=model_provider,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return model
    except Exception as e:
        return None


def create_rag_chain(prompt, model):
    if "reranking_retriever" not in st.session_state:
        loader = PyPDFDirectoryLoader("research_papers")
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )
        chunks = text_splitter.split_documents(documents)
        embedding = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        chroma = Chroma.from_documents(chunks, embedding)
        retriever = chroma.as_retriever(search_kwargs={"k": 6})
        reranker_model = HuggingFaceCrossEncoder(
            model_name="cross-encoder/ms-marco-MiniLM-L6-v2"
        )
        reranker = CrossEncoderReranker(model=reranker_model, top_n=3)
        st.session_state.reranking_retriever = ContextualCompressionRetriever(
            base_retriever=retriever, base_compressor=reranker
        )

    doc_chain = create_stuff_documents_chain(llm=model, prompt=prompt)
    retriever_chain = create_retrieval_chain(
        retriever=st.session_state.reranking_retriever, combine_docs_chain=doc_chain
    )

    return retriever_chain

def chatbot_func(
    model_provider, api_key, model_name, temperature, max_tokens, user_input
):
    system_prompt = """
    You are a question-answering assistant.

    Answer the user's question using the provided context.
    If the answer cannot be found in the context, say that you don't know.

    Context:
    {context}
    """

    prompt = ChatPromptTemplate.from_messages(
        [("system", system_prompt), ("human", "{input}")]
    )

    try:
        model = get_model(
            model_provider,
            api_key,
            model_name,
            temperature,
            max_tokens
        )

        if model is None:
            return {
                "answer": f"Cannot initialize this model {model_name}. Check your API key.",
                "context": []
            }

        rag_chain = create_rag_chain(prompt, model)

        response = rag_chain.invoke({
            "input": user_input
        })

        return {
            "answer": response["answer"],
            "context": response["context"]
        }

    except Exception as e:
        return {
            "answer": f"Error: {e}",
            "context": []
        }