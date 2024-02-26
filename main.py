# import os
import pandas as pd
import streamlit as st

#custome frontend modules
from animation import *
from Rightcolumn import *
from helperfunctions import *


st.set_page_config(layout="wide",initial_sidebar_state="expanded", page_title="LLM's Data Visualization", page_icon="⎍")
st.set_option('deprecation.showPyplotGlobalUse', False)

#header 
st.markdown("<h4 style='text-align: center; font-family:Menlo; color:orange; padding-top: 0rem;'>Data Visualization with LLM's</h4>", unsafe_allow_html=True)


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

    st.caption('''This Application is preloaded with the Diamonds and Pengiuns dataset available on Kaggle.
               The datasets are publically available  and can be examined via \n **[Diamonds](https://www.kaggle.com/shivam2503/diamonds)** & **[Penguins](https://www.kaggle.com/datasets/parulpandey/palmer-archipelago-antarctica-penguin-data)**.
               You can also upload your own dataset.
               ''')
    st.divider()

    on = st.toggle("Use own OpenAI API", False)
    if on:
        st.caption("Please provide the API keys for the LLM's")
        openai_key = st.text_input("OpenAI Key:", type="password")
        st.divider()
    else:
        st.caption("Currently using default API")
        # openai_key = os.environ["OPENAI_API_KEY"]
        st.divider()


    # Add facility to upload a dataset
    st.caption("Please upload CSV file formats only.Other formats are currently not supported.")
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

col = st.columns((6, 2), gap='medium')

with col[0]:
    question = st.text_area("Provide description for visualization", height=5)
    Vizbutton = st.button("Visualize Data",type="primary")

    if chosen_data:
        with st.expander("Dataset Preview"):
            st.write(dataframe[chosen_data])

    if Vizbutton:
        api_keys_entered = True

    # Check API keys are entered.
        if not openai_key.startswith('sk-') and on == True:
            st.error("Please enter a valid OpenAI API key.")
            api_keys_entered = False

        if on == False:
            api_keys_entered = True

        if api_keys_entered:
            code_generator(dataframe[chosen_data], question, openai_key)
            st.pyplot()
            insgight(dataframe[chosen_data], question, openai_key)
                                

with col[1]:
    setPrompt(dataframe[chosen_data])    # Add indented block of code here
