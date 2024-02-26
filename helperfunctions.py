from openai import OpenAI
# import os
import matplotlib.pyplot as plt
# from dotenv import load_dotenv, find_dotenv
import streamlit as st
from langchain_openai import ChatOpenAI

from langchain_experimental.agents import create_pandas_dataframe_agent

def generate_code(question_to_ask, api_key):
    # Set up OpenAI API key
    client = OpenAI(api_key=api_key,)
    
    # Request code generation from GPT-3.5
    task = "Generate Python Code Script.The script should only include code, no comments."

    response = client.chat.completions.create(model="gpt-3.5-turbo",
                                              stop=["plt.show()"],
                                              temperature=0,
                                              max_tokens=600,
                                              frequency_penalty=0,
                                              presence_penalty=0,
                                              top_p=1.0,
                                              messages=[{"role":"system","content":task},{"role":"user","content":question_to_ask}])
    
    # Extract the generated code from the response
    llm_response = response.choices[0].message.content
    generated = f"```python\n{llm_response}\n```"
    
    return generated

def code_generator(data, question, openai_key):

    # load_dotenv(find_dotenv(), override=True)

    # Now loading the llms
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.6, api_key=openai_key)
    panda_agent = create_pandas_dataframe_agent(llm, data, agent_type="openai-tools", verbose=True)

    code_visual = panda_agent.run(f"{question}  + generate python code to visualize the data that can be displayed in streamlit")
    code_visual = f"```python\n{code_visual}\n```"
    return code_visual

def insgight(data, question, openai_key):
    # load_dotenv(find_dotenv(), override=True)

    # Now loading the llms
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.5, api_key=openai_key)
    panda_agent = create_pandas_dataframe_agent(llm, data, agent_type="openai-tools", verbose=True)

    explanation = panda_agent.run(f"{question}  + issue verbal explanation only do not provide code")
    st.warning(f"**{explanation}**")



def code_generator2(data, question, openai_key):
    # load_dotenv(find_dotenv(), override=True)

    # Now loading the llms
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.6, api_key=openai_key)
    panda_agent = create_pandas_dataframe_agent(llm, data, agent_type="openai-tools", verbose=True)

    code_visual = panda_agent.run(f"{question}  + generate a st.altair_chart code to visualize the data that can be displayed in streamlit")

    # Parse the data name
    data_name = data.name if hasattr(data, 'name') else 'df'
    
    # Replace 'df' with the data name in the generated code
    code_visual = code_visual.replace('df', data_name)

    code_visual = f"```python\n{code_visual}\n```"
    
    # Display the Altair chart
    return code_visual

