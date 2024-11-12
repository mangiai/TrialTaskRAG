import chainlit as cl
import openai 
import langchain
import pinecone
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def read_doc(directory):
    file_loader = PyPDFDirectoryLoader(directory)
    documents = file_loader.load()
    if not documents:
        print(f"No documents found in the directory: {directory}")
    return documents

doc = read_doc('PDFs')
print(f"Number of documents loaded: {len(doc)}")

## Divide the docs into chunks
def chunk_data(docs, chunk_size=800, chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(docs)
    return chunks

documents = chunk_data(docs=doc)
print(f"Number of document chunks: {len(documents)}")

import os
import time
import getpass
from pinecone import Pinecone, ServerlessSpec

if not os.getenv("PINECONE_API_KEY"):
    os.environ["PINECONE_API_KEY"] = getpass.getpass("Enter your Pinecone API key: ")

pinecone_api_key = os.environ.get("PINECONE_API_KEY")

pc = Pinecone(api_key=pinecone_api_key)

from dotenv import load_dotenv
load_dotenv()

index_name = "raglegal"  

existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

if index_name not in existing_indexes:
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=os.getenv("PINECONE_ENV")),
    )
    while not pc.describe_index(index_name).status["ready"]:
        time.sleep(1)

index = pc.Index(index_name)

import getpass

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")  # Use a valid embedding model

from langchain_pinecone import PineconeVectorStore

vector_store = PineconeVectorStore(index=index, embedding=embeddings)

from uuid import uuid4
uuids = [str(uuid4()) for _ in range(len(documents))]
vector_store.add_documents(documents=documents, ids=uuids)

retriever = vector_store.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 1, "score_threshold": 0.5},
)

retriever.invoke("Briefly describe the Islamabad University Act", filter={"source": "news"})

from langchain_community.tools import TavilySearchResults

# Initialize the TavilySearchResults tool with custom settings
tavirly_search_tool = TavilySearchResults(
    max_results=5,  # Limit the number of search results
    search_depth="advanced",  # Use advanced search depth
    include_answer=True,  # Include an answer to the query in the results
    include_raw_content=True,  # Include the raw content from the search results
    include_images=True,  # Include image results
    name="Tavily_search",  # Custom name for the tool
    description="This tool searches the web using Tavirly and returns up to 5 results with advanced search depth, including raw content and images."  # Custom description
)

tools = [tavirly_search_tool.as_tool()]

import getpass
import os

if "GROQ_API_KEY" not in os.environ:
    os.environ["GROQ_API_KEY"] = getpass.getpass("Enter your Groq API key: ")

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    max_tokens=8000,  # You can adjust this as needed
    timeout=60,  # Set appropriate timeout based on complexity
    max_retries=2,
    # other params...
)

from langchain.prompts import PromptTemplate

prompt_template = """
You are an advanced AI legal assistant with access to multiple tools and a vector database of legal documents. Your task is to:

1. **Understand the user's legal query.**
2. **Retrieve relevant legal documents from the vector database if needed.**
3. **Use the appropriate tool to analyze, compare, or explain legal documents based on the context.**
4. **Synthesize the results and generate a final response that combines the vector store results and tool outputs.**

**Steps**:

- If the query involves **specific legal information or content search**, query the vector database for related documents.
- If the query involves **comparing legal documents**, retrieve the relevant documents and perform a detailed comparison highlighting similarities and differences.
- If the query requires **explaining legal provisions or clauses**, interpret the relevant sections and provide a clear, concise explanation.
- If the query involves **recent legal developments or case law**, use the Legal Research tool to find up-to-date information.

**Remember** to combine all relevant information in a cohesive and professional response, ensuring accuracy and clarity.

**Examples**:

- **Query**: "Compare the data protection regulations between Country A and Country B."
  - **Action**: Retrieve the relevant legal texts from the vector database, then analyze and compare the key differences and similarities in data protection laws between the two countries.
- **Query**: "Explain the implications of the new Cybersecurity Act."
  - **Action**: Retrieve the Act from the vector database and provide a summary explaining its main provisions and potential impact.
- **Query**: "What are the recent changes in employment law affecting overtime pay?"
  - **Action**: Use the Legal Research tool to find recent amendments or case law related to employment and overtime pay, then summarize the findings.

Now, proceed with the user's query.
"""

from langchain.agents import AgentType, initialize_agent
from langchain_community.chat_models import ChatOpenAI  # Ensure compatibility with ChatGroq

from langchain.memory import ConversationBufferMemory

# Initialize conversation memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Initialize the agent with memory
agent = initialize_agent(
    tools=tools,
    llm=llm,
    prompt=prompt_template,
    vector_store=vector_store,
    memory=memory,  # Pass memory here
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True,
    handle_parsing_errors=True,
)

@cl.on_chat_start
async def start():
    await cl.Message(content="Welcome to the Legal AI Assistant! How can I help you today?").send()

import json


import json

@cl.on_message
@cl.on_message
async def main(message: cl.Message):
    user_query = message.content  # Extract the string content

    # Run the agent with the user's query
    response = agent.invoke(user_query)

    # Check if response is a dictionary and has an 'output' key, otherwise use str(response)
    if isinstance(response, dict) and 'output' in response:
        pretty_response = response['output']
    else:
        # Fallback to converting response to a string if it doesn't match expected format
        pretty_response = str(response)

    # Send the cleaned response back to the user
    await cl.Message(content=pretty_response).send()
