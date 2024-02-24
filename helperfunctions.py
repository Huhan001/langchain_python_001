from langchain_openai import ChatOpenAI
from openai import OpenAI

def create_dataframe_primer(df_dataset, df_name):
    """
    Function to generate a primer description and code for visualizing a DataFrame.

    Parameters:
    - df_dataset: DataFrame object
    - df_name: Name of the DataFrame

    Returns:
    - primer_desc: Primer description as a string
    - primer_code: Python code for graphing the DataFrame
    """
    primer_desc = f"Using a DataFrame named '{df_name}' with columns: {', '.join(df_dataset.columns)}."
    
    for col in df_dataset.columns:
        if len(df_dataset[col].unique()) < 10 and df_dataset[col].dtype == "object":
            primer_desc += f"\nThe column '{col}' contains categorical values: {', '.join(map(str, df_dataset[col].drop_duplicates()))}."
            # primer_desc += f"\nThe column '{col}' contains categorical values: {', '.join(df_dataset[col].drop_duplicates())}."
        elif df_dataset[col].dtype in ["int64", "float64"]:
            primer_desc += f"\nThe column '{col}' is of type {df_dataset[col].dtype} and contains numeric values."

    primer_desc += "\nLabel the x and y axes appropriately."
    primer_desc += "\nAdd a title. Set the fig suptitle as empty."
    primer_desc += "{}"  # Space for additional instructions if needed
    primer_desc += "\nUsing Python version 3.9.12, create a script using the DataFrame to graph the following:"

    primer_code = (
        "import pandas as pd\n"
        "import matplotlib.pyplot as plt\n"
        "fig, ax = plt.subplots(1, 1, figsize=(10, 4))\n"
        "ax.spines['top'].set_visible(False)\n"
        "ax.spines['right'].set_visible(False)\n"
        f"df = {df_name}.copy()\n"
    )

    return primer_desc, primer_code


def chained_question(primer_desc, primer_code, question):
    """
    Format the primer description, question, and code.

    Parameters:
    - primer_desc: String, primer description
    - primer_code: String, code snippet for the primer
    - question: String, question related to the primer

    Returns:
    - formatted_text: Formatted text including primer description, question, and code
    """
    # Fill in the model-specific instructions variable
    instructions = "\nDo not use the 'c' argument in the plot function, use 'color' instead and only pass color names like 'green', 'red', 'blue'."

    # Add model-specific instructions to the primer description
    primer_desc = primer_desc.format(instructions)

    # Format the primer description, question, and code into a single string
    formatted_text = f'"""\n{primer_desc}\n{question}\n"""\n{primer_code}'
    
    return formatted_text

def generate_code(question_to_ask, api_key):
    # Set up OpenAI API key
    client = OpenAI(api_key=api_key,)
    
    # Request code generation from GPT-3.5
    task = """Generate Python Code Script.
            The script should only include code, no comments."""

    response = client.chat.completions.create(model="gpt-3.5-turbo",
                                              messages=[{"role":"system","content":task},{"role":"user","content":question_to_ask}])
    
    # Extract the generated code from the response
    # llm_response = response["choices"][0]["message"]["content"]
    llm_response = response.choices[0].message.content
    
    # Remove unnecessary lines related to 'read_csv'
    generated_code = remove_read_csv_lines(llm_response)
    
    return generated_code

def remove_read_csv_lines(code):
    # Find the position of 'read_csv'
    csv_position = code.find("read_csv")
    
    # Remove lines containing 'read_csv'
    if csv_position > 0:
        line_before_csv = code[:csv_position].rfind("\n")
        line_after_csv = code[csv_position:].find("\n")
        
        if line_before_csv == -1:
            code_before = ""
        else:
            code_before = code[:line_before_csv]
        
        if line_after_csv == -1:
            code_after = ""
        else:
            code_after = code[csv_position + line_after_csv:]
        
        code = code_before + code_after
    
    return code
