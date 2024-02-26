import os
from dotenv import load_dotenv, find_dotenv
import streamlit as st
from langchain_openai import ChatOpenAI

from langchain_experimental.agents import create_pandas_dataframe_agent

def setPrompt(data):

    load_dotenv(find_dotenv(), override=True)

    # Now loading the llms
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.6, api_key=os.environ["OPENAI_API_KEY"])
    panda_agent = create_pandas_dataframe_agent(llm, data, agent_type="openai-tools", verbose=True)


    st.write("**Data Overview**")
    missing_values = panda_agent.run("how many missing values?")
    st.info(f"**{missing_values}**")
    duplicates = panda_agent.run("how many duplicates?")
    st.info(f"**{duplicates}**")
    insights = panda_agent.run("briefly describe the data?")
    st.success(f"**{insights}**")
    drawback = panda_agent.run("conjour only one tactical decision from the data?")
    # st.warning(f"**{drawback}**")
    # st.write("___")


def code_generator(data, question, openai_key):

    load_dotenv(find_dotenv(), override=True)

    # Now loading the llms
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.6, api_key=openai_key)
    panda_agent = create_pandas_dataframe_agent(llm, data, agent_type="openai-tools", verbose=True)


    code_visual = panda_agent.run(f"{question}  + generate python code to visualize the data that can be displayed in streamlit")
    code_visual = f"```python\n{code_visual}\n```"
    return code_visual

def insgight(data, question, openai_key):
    load_dotenv(find_dotenv(), override=True)

    # Now loading the llms
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.5, api_key=openai_key)
    panda_agent = create_pandas_dataframe_agent(llm, data, agent_type="openai-tools", verbose=True)

    explanation = panda_agent.run(f"{question}  + issue verbal explanation of only")
    st.warning(f"**{explanation}**")