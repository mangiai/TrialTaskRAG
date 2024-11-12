# Gaditek Legal System Documentation  
*Submission for NLP Task*

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
