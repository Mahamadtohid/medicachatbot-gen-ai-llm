from src.helper import load_pdf , text_split , download_hugging_face_embeddingd
from langchain.vectorstores import Pinecone 
import pinecone 
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.environ.get('PINECONE_API_KEY')
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV')


extracted_data = load_pdf("data/")
text_chunks= text_split(extracted_data)

embeddings = download_hugging_face_embeddingd()
    
import os
serverless =os.environ.get('SERVERLESS') or True

from pinecone import Pinecone
api_key = os.environ.get('PINECONE_API_KEY') or PINECONE_API_KEY
pc = Pinecone(api_key = PINECONE_API_KEY)

from pinecone import ServerlessSpec, PodSpec

if serverless:
    spec = ServerlessSpec(cloud="aws" , region="us-east-1")
    
else:
    spec = PodSpec(environment="us-east-1")

index_name = 'medical-chatbot'

import time

existing_indexes =[
    index_info["name"] for index_info in pc.list_indexes()
]
if index_name not in existing_indexes:
    pc.create_index(
        index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(
            cloud='aws', 
            region='us-east-1'
        ) 
    ) 
    
    while not pc.describe_index(index_name).status['ready']:
        time.sleep(1)
        
index =pc.Index(index_name)
time.sleep(1)

index.describe_index_stats()

{
    'dimension': 384 ,
    'index_fullness' :0.0,
    'namespaces' :{'':{'vector_count':26781}},
    'total_vector_count' :26781
}




from langchain.vectorstores import Pinecone as PineconeStore
import langchain_pinecone
from langchain import *
import os
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
os.environ['PINECONE_API_KEY'] = '**************************************'
docsearch = langchain_pinecone.PineconeVectorStore.from_texts([t.page_content for t in text_chunks],embeddings,index_name="medical-chatbot")
