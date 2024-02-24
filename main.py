import openai
import os
import pandas as pd
from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_community.llms import OpenAI
from langchain.chains import LLMChain
import streamlit as st
from animation import *

#header 
st.markdown("<h4 style='text-align: center; font-family:Menlo; color:orange; padding-top: 0rem;'>Data Visualization with LLM's</h4>", unsafe_allow_html=True)
# st.markdown("<h3 style='text-align: center; font-weight:bold; font-family:comic sans ms; padding-top: 0rem;'>Data Visualization with LLM's</h3>", unsafe_allow_html=True)


if "dataframe" not in st.session_state:
    dataframe = {}
    # Preloaded data
    dataframe["Diamonds"] = pd.read_csv("diamonds.csv")
    dataframe["Penguins"] = pd.read_csv("penguins.csv")
    st.session_state["dataframe"] = dataframe
else:
    # use uploaded dataframe
    dataframe = st.session_state["dataframe"]

with st.sidebar:

    st.write("Harnessing LLM's for Data Visualization")

    st.caption('''The Application is preloaded with the Diamonds and Pengiuns dataset available on Kaggle.
               The datasets are publically available  and can be examined via \n **[Diamonds](https://www.kaggle.com/shivam2503/diamonds)** & **[Penguins](https://www.kaggle.com/datasets/parulpandey/palmer-archipelago-antarctica-penguin-data)**.
               You can also upload your own dataset.
               ''')
    st.caption("Please upload CSV file formats only.Other formats are currently not supported.")
    st.divider()

    # Add facility to upload a dataset
    upload = st.file_uploader("Upload CSV file:", type="csv")
    
    # When we add the radio buttons we want to default the selection to the first
    index_no = 0

    if upload:
        # Read in the data, add it to the list of available datasets. Give it a nice name.
        Name = upload.name[:-4].capitalize()
        dataframe[Name] = pd.read_csv(upload)
        index_no = len(dataframe) - 1

    # First we want to choose the dataset, but we will fill it with choices once we've loaded one
    dataframe_container = st.empty()

    # Radio buttons for dataset choice
    chosen_data = dataframe_container.radio("Choose a dataset to Exploit:", list(dataframe.keys()), index=index_no)
    golive()