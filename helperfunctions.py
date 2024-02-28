from openai import OpenAI
# import os
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from langchain_openai import ChatOpenAI

from langchain_experimental.agents import create_pandas_dataframe_agent


# 📌
def generate_code(prompt_question, api_key):
    # Set up OpenAI API key
    client = OpenAI(api_key=api_key,)
    
    # Request code generation from GPT-3.5
    task = "Generate Python Code Script.The script should only include code, no comments."

    response = client.chat.completions.create(model="gpt-3.5-turbo",
                                              temperature=0,
                                              max_tokens=600,
                                              frequency_penalty=0,
                                              presence_penalty=0,
                                              top_p=1.0,
                                              messages=[{"role":"system","content":task},{"role":"user","content":prompt_question}])
    
    # Extract the generated code from the response
    llm_response = response.choices[0].message.content
    removed_csv = remove_csv_occurence(llm_response)
    
    return removed_csv


# 📌
def code_generator(data, question, openai_key):

    # load_dotenv(find_dotenv(), override=True)

    # Now loading the llms
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.6, api_key=openai_key)
    panda_agent = create_pandas_dataframe_agent(llm, data, agent_type="openai-tools", verbose=True)

    code_visual = panda_agent.run(f"{question}  + generate python code to visualize the data that can be displayed in streamlit")
    code_visual = f"```python\n{code_visual}\n```"
    return code_visual


# 📌
def insgight(data, question, openai_key):
    # load_dotenv(find_dotenv(), override=True)

    # Now loading the llms
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.5, api_key=openai_key)
    panda_agent = create_pandas_dataframe_agent(llm, data, agent_type="openai-tools", verbose=True)

    explanation = panda_agent.run(f"{question}  + issue verbal explanation only do not provide code")
    st.warning(f"**{explanation}**")


# 📌
def split_question_explanations(df_dataset, df_name, question):

    """
    This function takes a dataframe along with its name
    and a list of column names. For each column provided,
    it identifies those with fewer than 15 unique values,
    implying potential categorical variables. These columns' values
    are added to the primary axis, ensuring horizontal grid lines
    and proper labeling for enhanced visualization.
    """

    description = f"Use a dataframe called {df_name} with columns '{','.join(df_dataset.columns)}'. "
    
    for i in df_dataset.columns:
        unique_values = df_dataset[i].drop_duplicates()
        if len(unique_values) < 15 and df_dataset.dtypes[i] == "object":
            description += f"\nThe column '{i}' has categorical values '{','.join(map(str, unique_values))}'. "
        elif df_dataset.dtypes[i] in ["int64", "float64"]:
            description += f"\nThe column '{i}' is type {df_dataset.dtypes[i]} and contains numeric values. "
            
    description += "\nLabel the x and y axes appropriately."
    description += "\nAdd a title. Set the fig suptitle as empty."
    description += "{}"  # Space for additional instructions if needed
    description += "\nUsing Python version 3.9.12, create a script using the dataframe df to graph the following: "
    
    codebase = "import pandas as pd\nimport matplotlib.pyplot as plt\n"
    codebase += "fig, ax = plt.subplots(1, 1, figsize=(10, 4))\n"
    codebase += "ax.spines['top'].set_visible(False)\nax.spines['right'].set_visible(False)\n"
    codebase += f"{df_name} = df_dataset.copy()\n"
    
    currate_question = '"""\n' + description + question + '\n"""\n' + codebase
    return currate_question


# 📌
def remove_csv_occurence(code):
    csv_line = code.find("read_csv")
    if csv_line > 0:
        return_before_csv_line = code[:csv_line].rfind("\n")
        code = code[:return_before_csv_line] + code[csv_line + code[csv_line:].find("\n"):]
    return code


# 📌
def repurposed_split_question(df_dataset, df_name):
    description = f"Use a dataframe called {df_name} with columns '{','.join(df_dataset.columns)}'. "
    
    for i in df_dataset.columns:
        unique_values = df_dataset[i].drop_duplicates()
        if len(unique_values) < 15 and df_dataset.dtypes[i] == "object":
            description += f"\nThe column '{i}' has categorical values '{','.join(map(str, unique_values))}'. "
        elif df_dataset.dtypes[i] in ["int64", "float64"]:
            description += f"\nThe column '{i}' is type {df_dataset.dtypes[i]} and contains numeric values. "
            
    description += "\nLabel the x and y axes appropriately."
    description += "\nAdd a title. Set the fig suptitle as empty."
    description += "{}"  # Space for additional instructions if needed
    description += "\nUsing Python version 3.9.12, create a script using the dataframe df to graph the following: "
    
    codebase = "import pandas as pd\nimport matplotlib.pyplot as plt\n"
    codebase += "fig, ax = plt.subplots(1, 1, figsize=(10, 4))\n"
    codebase += "ax.spines['top'].set_visible(False)\nax.spines['right'].set_visible(False)\n"
    codebase += f"{df_name} = df_dataset.copy()\n"
    
    
    return description, codebase


# 📌
def replace_df_copy_with_chosen_data_copy(code, chosen_data):
    # Replace occurrences of "df_dataset.copy()" with "dataframe[chosen_data].copy()"
    modified_code = code.replace("df_dataset.copy()", f"dataframe['{chosen_data}'].copy()")
    return modified_code

