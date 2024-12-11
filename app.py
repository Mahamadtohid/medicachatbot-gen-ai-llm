from flask import Flask , render_template , request
from waitress import serve
from src.helper import download_hugging_face_embeddingd
from langchain.vectorstores import Pinecone  
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from langchain import *
from langchain_pinecone import PineconeVectorStore
from src.prompt import *
from langchain.chains import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from pydantic import Field
from langchain.callbacks.manager import CallbackManagerForRetrieverRun 
from langchain.callbacks.manager import CallbackManagerForRetrieverRun, AsyncCallbackManagerForRetrieverRun
from typing import *
from langchain.schema.retriever import BaseRetriever, Document
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
import os

app = Flask(__name__)

load_dotenv()

api_key = os.environ.get('PINECONE_API_KEY')
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

embeddings = download_hugging_face_embeddingd()

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

docsearch = PineconeVectorStore.from_existing_index(index_name,embeddings)

PROMPT=PromptTemplate(template=prompt_template, input_variables=["context","question"])
chain_type_kwargs={"prompt":PROMPT}

llm = CTransformers(model="model/llama-2-7b-chat.ggmlv3.q4_0.bin",
                    model_type="llama",
                    config={'max_new_tokens':512,
                            'temperature':0.8})


from langchain.chains import LLMChain

llm_chain = LLMChain(llm=llm, prompt=PROMPT)

custom_chain = StuffDocumentsChain(
    llm_chain=llm_chain, 
    document_variable_name="context"
)

class PineconeRetriever(BaseRetriever):
    vector_store: PineconeVectorStore
    k: int = Field(default=2)

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
        return self.vector_store.similarity_search(query, k=self.k)

    async def _aget_relevant_documents(self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun) -> List[Document]:
        return await self.vector_store.asimilarity_search(query, k=self.k)

retriever = PineconeRetriever(vector_store=docsearch, k=3)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt": PROMPT}
)


@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET" , "POST"])
def chat():
    msg = request.form["msg"]
    input =msg
    print(input)
    
    result=qa({"query":input})
    print("Response : " , result["result"])
    return str(result['result'])


if __name__ == '__main__':
    if app.debug:
        app.run( host="0.0.0.0", port=8081)
    else:
        serve(app, host="0.0.0.0", port=8081)

    
    
