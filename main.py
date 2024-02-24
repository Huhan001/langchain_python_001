import openai
import os
import pandas as pd
from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_community.llms import OpenAI
from langchain.chains import LLMChain
import streamlit as st
import altair as alt
from langchain_experimental.agents import create_pandas_dataframe_agent

#custome frontend modules
from animation import *
from Rightcolumn import *
from helperfunctions import *


st.set_page_config(layout="wide",initial_sidebar_state="expanded")
plot_area = st.empty()

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
        openai_key = os.environ["OPENAI_API_KEY"]
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
    question = st.text_area("Provide description for visualization", height=9)
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

        # if api_keys_entered:
        # # Get the primer for the chosen dataset
        #     primer_desc, primer_code = create_dataframe_primer(dataframe[chosen_data], chosen_data)

        #     try:
        #         # Format the question
        #         question = chained_question(primer_desc, primer_code, question)

        #         # Generate code using OpenAI API
        #         generate_code = ""
        #         generated_code = generate_code(question, openai_key)

        #         # Execute the generated code
        #         generated_code = generated_code + primer_code
        #         print(generated_code)
        #         st.write("Graph Visualization")
        #         plot_area = st.empty()
        #         plot_area.pyplot(exec(generated_code))

        #     except Exception as e:
        #         if type(e) == openai.APIConnectionError:
        #             st.error("The server could not be reached")
        #         elif type(e) == openai.APIStatusError:
        #             st.error("Another non-200-range status code was received")
        #         elif type(e) == openai.RateLimitError:
        #             st.error("A 429 status code was received; we should back off a bit.")
        if api_keys_entered:
    # Get the primer for the chosen dataset
            primer_desc, primer_code = create_dataframe_primer(dataframe[chosen_data], chosen_data)

            try:
                # Format the question
                question = chained_question(primer_desc, primer_code, question)

                # Generate code using OpenAI API
                generated = ""
                generated = generate_code(question, openai_key)

                # Execute the generated code
                generated = primer_code + generated
        
                st.write("Graph Visualization")
                plot_area.pyplot(exec(generated))

            except Exception as e:
                if type(e) == openai.APIConnectionError:
                    st.error("The server could not be reached")
                elif type(e) == openai.APIStatusError:
                    st.error("Another non-200-range status code was received")
                elif type(e) == openai.RateLimitError:
                    st.error("A 429 status code was received; we should back off a bit.")

                

with col[1]:
    setPrompt(dataframe[chosen_data])    # Add indented block of code here
