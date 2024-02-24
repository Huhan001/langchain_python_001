import os
from dotenv import load_dotenv, find_dotenv
from langchain.schema import (
        AIMessage, # message from the AI
        HumanMessage, # message from the human
        SystemMessage  #systme message from  openai API
    )
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
# from langchain.llms import OpenAI
from langchain_community.llms import OpenAI
from langchain.chains import LLMChain

    

def setup(provide):
    load_dotenv(find_dotenv(), override=True)
    #finddotenv() will look for .env file in the current directory and all the parent directories
    print(os.getenv(provide))

def setSchema(provide):
    # Create a schema
    chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.5, max_tokens=1024, openai_api_key=os.getenv(provide))
    messages = [
        SystemMessage(content='you are a physicist and respond in swahili'),
        HumanMessage(content='explain quantom mechanixs in one sentence')
    ]
    output = chat(messages)
    print(output.content)


def setPrompt():
    template = ''' you are an economist give an overview  on {topic} in {language}.'''

    prompt = PromptTemplate(
        input_variable=['topic', 'language'],
        template=template)
    
    #now loading the llms
    llm = OpenAI(model_name="gpt-3.5-turbo", temperature=0.7, max_tokens=1024, openai_api_key=os.getenv('openai_api'))
    feedback = llm(prompt.format(topic = 'economics', language = 'french'))
    print(feedback.content)

# simple chains are essentially a list of functions that are called in sequence
# each function takes a message and returns a message
# the output of the previous function is the input of the next
    
def simple_chain(provide):
    chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.5, max_tokens=1024, openai_api_key=os.getenv(provide))
    template = ''' you are an economist give an overview  on {topic} in {language}.'''
    prompt = PromptTemplate(
        input_variable=['topic', 'language'],
        template=template)
    
    chain = LLMChain(prompt=prompt, llm=chat)
    output = chain.run({'topic': 'international trade', 'language': 'french'})
    print(output)

def Usingagents():
    from langchain_experimental.agents.agent_toolkits import create_python_agent
    from langchain_experimental.tools.python.tool import PythonAstREPLTool

    agent_executor = create_python_agent(
        llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.5, max_tokens=1024, openai_api_key=os.getenv('openai_api')),
        tool=PythonAstREPLTool(),
        verbose=True # print the output of the agent
    )
    agent_executor.run('calculate the square root of factorial 20 and display the result in a graph.')


def textsplitter():
    load_dotenv(find_dotenv(), override=True)
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=10, # the number of characters to overlap between chunks
        length_function=len # the function to use to get the length of the text
    )
    with open('sourceforchat.txt', 'r') as file:
        text = file.read()
    
    chunks = splitter.create_documents([text])

    # embeddings
    # from langchain_community.embeddings import OpenAIEmbeddings
    from langchain_openai import OpenAIEmbeddings
    embeddings = OpenAIEmbeddings(model_name="gpt-3.5-turbo", openai_api_key=os.getenv('openai_api'))
    # vector = embeddings.embed_query(chunks[0])
    # print(vector)

    #initializign pinecone client
    import pinecone
    from langchain.vectorstores import Pinecone 

    pinecone.init(api_key=os.getenv('pinecone_api'))

    #delete all indexes 
    for index in pinecone.list_indexes():
        print('deleting all indexes')
        pinecone.delete_index(index)
        print('all indexes deleted')
    
    #create a new index
    index_name = 'random-index'
    if index_name not in pinecone.list_indexes():
        print('creating index {}'.format(index_name))
        pinecone.create_index(name=index_name, metric='cosine', dimension=153)
        print('index created')

    


import os 
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)


def load_document(file):
    _, extension = os.path.splitext(file)
    if extension == ".pdf":
        from langchain_community.document_loaders import PyPDFLoader
        print("Loading document {}".format(file))
        loader = PyPDFLoader(file)
    elif extension == ".docx":
        from langchain_community.document_loaders import Docx2txtLoader
        print("Loading document {}".format(file))
        loader = Docx2txtLoader(file)
    elif extension == ".txt":
        from langchain_community.document_loaders import TxtLoader
        print("Loading document {}".format(file))
        loader = TxtLoader(file)
    elif extension == ".csv":
        from langchain_community.document_loaders import CSVLoader
        print("Loading document {}".format(file))
        loader = CSVLoader(file)

    data = loader.load()

    # chunking
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(chunk_size=256, chunk_overlap=0, 
                                              separators=['. ', '\n', '\r\n', '!', '?', ';', ':','/', '(', ')', '[', ']',',','\t'],
                                              is_separator_regex=True)
    chunks = splitter.split_documents(data)


    # import pinecone
    # from langchain_community.vectorstores import Pinecone
    # from langchain_openai import OpenAIEmbeddings

    # # Load the embeddings
    # embeddings = OpenAIEmbeddings(model_name="gpt-3.5-turbo", openai_api_key=os.getenv('openai_api'))
    # pinecone.init(api_key=os.getenv('pinecone_api'))

    # # Create a new Pinecone index
    # index_name = "langchain"
    # if index_name not in pinecone.list_indexes():
    #     pinecone.create_index(index_name, metric="cosine", shards=1, dimension=256)
    #     vector_store = Pinecone.from_documents(index_name, chunks, embeddings)
    #     print("Index created")
    #     # return vector_store
    # else:
    #     print("Index already exists")

    # get answers
    from langchain.chains import RetrievalQA
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=os.getenv('openai_api'))
    retriver = chunks.as_retriever(search_type="similarity", search_kwargs={'k': 3})
    chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriver, chain_type="stuff")
    answers = chain.run("What is the salary of level 9?")
    print(answers)

    # ask with memory
    