from dotenv import load_dotenv
from langchain_classic.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser

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


def chatbot_func(
    model_provider, api_key, model_name, temperature, max_tokens, user_input
):
    system_prompt = """
        you are an Question and answer assistant help user with reasnable answer 
        """
    prompt = ChatPromptTemplate.from_messages(
        [("system", system_prompt), ("human", "{input}")]
    )
    try:
        model = get_model(model_provider, api_key, model_name, temperature, max_tokens)
        if model is None:
            return f"Cannot initialize this model {model_name} check you api key "
        output_parser = StrOutputParser()
        chain = prompt | model | output_parser
        return chain.invoke({"input": user_input})
    except Exception as e:
        return f"Error: {e}"
