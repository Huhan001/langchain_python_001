import os
from dotenv import load_dotenv, find_dotenv
from langchain.schema import (
        AIMessage, # message from the AI
        HumanMessage, # message from the human
        SystemMessage  #systme message from  openai API
    )
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain

    

def setup(provide):
    load_dotenv(find_dotenv(), override=True)
    #finddotenv() will look for .env file in the current directory and all the parent directories
    print(os.getenv(provide))

def setSchema(provide):
    # Create a schema
    chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.5, max_tokens=1024, openai_api_key=os.getenv(provide))
    messages = [
        SystemMessage(content='you are a physicist and respond in swahili'),
        HumanMessage(content='explain quantom mechanixs in one sentence')
    ]
    output = chat(messages)
    print(output.content)


def setPrompt():
    template = ''' you are an economist give an overview  on {topic} in {language}.'''

    prompt = PromptTemplate(
        input_variable=['topic', 'language'],
        template=template)
    
    #now loading the llms
    llm = OpenAI(model_name="gpt-3.5-turbo", temperature=0.7, max_tokens=1024, openai_api_key=os.getenv('openai_api'))
    feedback = llm(prompt.format(topic = 'economics', language = 'french'))
    print(feedback.content)

# simple chains are essentially a list of functions that are called in sequence
# each function takes a message and returns a message
# the output of the previous function is the input of the next
    
def simple_chain(provide):
    chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.5, max_tokens=1024, openai_api_key=os.getenv(provide))
    template = ''' you are an economist give an overview  on {topic} in {language}.'''
    prompt = PromptTemplate(
        input_variable=['topic', 'language'],
        template=template)
    
    chain = LLMChain(prompt=prompt, llm=chat)
    output = chain.run({'topic': 'international trade', 'language': 'french'})
    print(output)