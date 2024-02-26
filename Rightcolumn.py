# import os
# from dotenv import load_dotenv, find_dotenv
import streamlit as st
from langchain_openai import ChatOpenAI

from langchain_experimental.agents import create_pandas_dataframe_agent

# 📌
def setPrompt(data):

    # load_dotenv(find_dotenv(), override=True)

    # Now loading the llms
    # llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.6, api_key=os.environ["OPENAI_API_KEY"])
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.6, api_key=st.secrets["OPENAI_API_KEY"])
    panda_agent = create_pandas_dataframe_agent(llm, data, agent_type="openai-tools", verbose=True)


    st.write("**Data Overview**")
    missing_values = panda_agent.run("how many missing values?")
    st.info(f"**{missing_values}**")
    duplicates = panda_agent.run("how many duplicates?")
    st.info(f"**{duplicates}**")
    insights = panda_agent.run("briefly describe the data?")
    st.success(f"**{insights}**")
    drawback = panda_agent.run("conjour only one tactical decision from the data?")
    st.warning(f"**{drawback}**")
    
    st.markdown("<h4 style='text-align: center; font-family:Menlo; padding-top: 0rem;'>🕊️</h4>", unsafe_allow_html=True)