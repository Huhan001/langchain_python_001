from openai import OpenAI

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


