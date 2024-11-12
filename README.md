# LegalQuery Documentation    
*Submission for Trial Assignment @ Gaditek*

## Project Overview
This project implements a Retrieval-Augmented Generation (RAG) system designed to handle legal queries by accessing a vectorized database of legal documents. By combining LangChain, Pinecone, and a language model, the system provides accurate responses based on contextually relevant legal information.

## Objectives
1. **Data Collection**: Gather legal documents from official sources and convert them into a searchable format.
2. **Text Processing and Embedding**: Process documents into chunks, generate embeddings, and store them in a vector database for efficient retrieval.
3. **Query Processing and Retrieval**: Accept user queries, retrieve relevant document chunks, and use them as context for generating responses.
4. **User Interface**: Utilize Chainlit to allow users to interact with the system in a straightforward and intuitive way.

## Code Walkthrough
### 1. Importing Libraries  
This RAG system relies on several libraries, such as:
- OpenAI and LangChain for embedding generation and language model interaction.
- Pinecone for vector database storage and retrieval.
- PyPDFDirectoryLoader from `langchain_community.document_loaders` to load and process PDF documents.
- RecursiveCharacterTextSplitter from `langchain.text_splitter` for text chunking.

```python
import openai
import langchain
import pinecone
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
```

### 2. Loading Environmental Variables
To securely manage API keys and environment settings, environment variables are loaded to ensure sensitive information is not hard-coded.
Create a .env file with these content
```bash
PINECONE_API_KEY="your_pinecone_api_key"
OPENAI_API_KEY="youre_openai_api_key"
TAVILY_API_KEY="your_tavily_api_key"
```


```bash
from dotenv import load_dotenv
load_dotenv()
```


### 3. Reading Documents Helper Function

The read_doc function reads PDF files from a specified directory and converts them into a format LangChain can process.

```python
def read_doc(directory):
    file_loader = PyPDFDirectoryLoader(directory)
    documents = file_loader.load()
    return documents

# Load documents
doc = read_doc('Law Dataset')
len(doc)
```

### 4. Chunking Documents
Documents are split into smaller, semantically meaningful chunks for retrieval efficiency using RecursiveCharacterTextSplitter.

```python
def chunk_data(docs, chunk_size=800, chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    doc = text_splitter.split_documents(docs)
    return docs

# Apply chunking
documents = chunk_data(docs=doc)
len(documents)
```
### 5. Setting up Pinecone Index
A Pinecone index is created to store embeddings, defining index parameters such as dimensionality and similarity metric.
```python
import os
import time
import getpass
from pinecone import Pinecone, ServerlessSpec

# Set up Pinecone API key
if not os.getenv("PINECONE_API_KEY"):
    os.environ["PINECONE_API_KEY"] = getpass.getpass("Enter your Pinecone API key:")

pinecone_api_key = os.environ.get("PINECONE_API_KEY")
pc = Pinecone(api_key=pinecone_api_key)

# Define index
index_name = "raglegal"
existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

# Create index if it doesn't exist
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
```
### 6. Embedding and Vector Store Setup
Document chunks are converted to embeddings using OpenAI’s embedding model and stored in Pinecone’s vector database.
```python
import getpass
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
from langchain_pinecone import PineconeVectorStore

vector_store = PineconeVectorStore(index=index, embedding=embeddings)
```
### 7. Adding Documents to Vector Store
Each document chunk is assigned a unique ID and stored in Pinecone for retrieval.
```python
from uuid import uuid4

# Assign unique IDs and add to vector store
uuids = [str(uuid4()) for _ in range(len(documents))]
vector_store.add_documents(documents=documents, ids=uuids)
```
### 8. Setting Up Vector Store Retriever
The retriever fetches chunks based on similarity to the user’s query.

```python
retriever = vector_store.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 1, "score_threshold": 0.5},
)

# Example query
retriever.invoke("Briefly describe the Islamabad University Act", filter={"source": "news"})
```
### 9. Setting Up Tavirly Tool for Additional Context Retrieval
The Tavirly tool allows external web search to supplement in-database legal information with recent developments.
```python
from langchain_community.tools import TavilySearchResults

tavirly_search_tool = TavilySearchResults(
    max_results=5,
    search_depth="advanced",
    include_answer=True,
    include_raw_content=True,
    include_images=True,
    name="Tavily_search",
    description="This tool searches the web using Tavirly and returns up to 5 results with advanced search depth, including raw content and images."
)

tools = [tavirly_search_tool.as_tool()]
```
### 10. Setting Up the Language Model (LLM)
ChatOpenAI is configured as the language model to generate responses based on retrieved context.

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="gpt-4",
    temperature=0,
    max_tokens=8000,
    timeout=60,
    max_retries=2,
)
```
### 11. Adding Prompt Template
A prompt template guides the model on how to handle legal queries, including retrieving relevant documents and providing legal explanations.
```python
from langchain.prompts import PromptTemplate

prompt_template = """
You are an advanced AI legal assistant with access to multiple tools and a vector database of legal documents. Your task is to:
- Understand the user's legal query.
- Retrieve relevant documents, analyze, compare, or explain them as needed.
- Generate a cohesive, professional response.

Examples:
- Query: "Compare data protection regulations between Country A and Country B."
- Query: "Explain the implications of the new Cybersecurity Act."
Now, proceed with the user's query.
"""
```
### 12. Creating the ReAct Agent
The ReAct agent integrates the tools, vector store, and prompt template with the LLM, enabling it to retrieve and analyze documents based on user queries.
```python
from langchain.agents import AgentType, initialize_agent

agent = initialize_agent(
    tools=tools,
    llm=llm,
    prompt=prompt_template,
    vector_store=vector_store,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True,
    handle_parsing_errors=True,
)

# Example query
query = "Briefly Describe the Islamabad University Act?"
response = agent.run(query)

# Output the response
print("Agent Response:")
print(response)
```
### 13. Lock and Freeze Requirements
Export environment requirements to ensure all dependencies are locked.
```bash
%pip freeze > requirement.txt
```


## Run Locally

Clone the project

```bash
  git clone https://github.com/mangiai/TrialTaskRAG.git
```

Go to the project directory

## Set up a Virtual Environment
Create a virtual environment named myenv (Linux):

```bash
python3 -m venv myenv
```
Activate the virtual environment:
```bash
source myenv/bin/activate
```
## Install Dependencies
Install the required packages:

```bash
pip install -r requirements.txt
```
## Set up Environment Variables
Ensure your .env file with API keys is in the project root directory, as shown below:
```plaintext
PINECONE_API_KEY="your_pinecone_api_key"
OPENAI_API_KEY="your_openai_api_key"
TAVILY_API_KEY="your_tavily_api_key"
```
## Start the Chainlit Server
Run chat.py with Chainlit:
```bash
chainlit run chat.py -w
```
The server should now be running locally, and you can access the interface by navigating to the URL provided by Chainlit (usually http://localhost:8000).












