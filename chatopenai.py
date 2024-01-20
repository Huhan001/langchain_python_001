import os
from dotenv import load_dotenv, find_dotenv
from datetime import datetime, date, time, timedelta

#load openai
import openai

def functionTorun():
    _ = load_dotenv(find_dotenv())
    openai.api_key = os.environ['OPENAI_API_KEY']
    # print(openai.api_key)

    currentDT = datetime.now().date()
    target_date = date(2024, 6, 12)

    if currentDT > target_date:
        llm_model = "gpt-3.5-turbo"
        return llm_model
    else:
        llm_model = "gpt-3.5-turbo-0301"
        return llm_model
    

def open_ai_function(prompt):

    model = functionTorun()
    message = [{"role": "user", "text": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=message,
        max_tokens=150,
    )
    print(response)



    
customer_email = """
                    Arrr, I be fuming that me blender lid \
                    flew off and splattered me kitchen walls \
                    with smoothie! And to make matters worse,\
                    the warranty don't cover the cost of \
                    cleaning up me kitchen. I need yer help \
                    right now, matey!
                    """
style = """American English \
            in a calm and respectful tone
            """
    
prompt = f"""Translate the text \
            that is delimited by triple backticks 
            into a style that is {style}.
            text: ```{customer_email}```
            """
