from django.shortcuts import render
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
#database
import io
from django.db import connections
import psycopg2
import faiss
import pickle
from io import BytesIO
import tempfile
from langchain.vectorstores.pgvector import PGVector


# Create your views here.
# def index(request):
#     return render(request, 'i.html') 

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_docs):
    text=""
    with io.BytesIO(pdf_docs.read()) as pdf_file:
        pdf_reader = PdfReader(pdf_file)
    # for pdf in pdf_docs:
    #     pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return  text



def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


# def get_vector_store(text_chunks):
    # embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    # vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    # vector_store.save_local("faiss_index")

    # # Get the default Django database connection
    # CONNECTION_STRING = connections['default']
    # CONNECTION_NAME = 'state_of_union_vectors'
    
    # chunks = index()
    # db = PGVector.from_documents(embedding=embeddings, documents=chunks, collection_name=CONNECTION_NAME, connection_string = CONNECTION_STRING)


def get_conversational_chain():

    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro",
                             temperature=0.3)

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain



def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    
    new_db = FAISS.load_local("faiss_index", embeddings)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    
    response = chain(
        {"input_documents":docs, "question": user_question}
        , return_only_outputs=True)

    print(response)
    response = response["output_text"].replace("**", "\n")
    return response


def index(request):
    if request.method == "POST" and request.FILES['upload']:
        if 'upload' not in request.FILES:
            err = 'No Images Selected'
            return render(request, 'index.html', {'err': err})
        f = request.FILES['upload']
        
        if f == '':
            err = 'No Files Selected'
            return render(request, 'index.html', {'err': err})
        
        user_question = request.POST.get('query')
    
        raw_text = get_pdf_text(f)
        text_chunks = get_text_chunks(raw_text)
        
        #chunks with page_content
        # chunks = [{"page_content": chunk} for chunk in  text_chunks] 
        
        # Create Faiss index and save it
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("faiss_index")

        # connection_string = connections['default'].settings_dict['ENGINE']
        DATABASES = connections['default'].settings_dict
        connection_string=f"postgresql://{DATABASES['USER']}:{DATABASES['PASSWORD']}@{DATABASES['HOST']}:{DATABASES['PORT']}/{DATABASES['NAME']}"
        
        text_embeddings = embeddings.embed_documents(text_chunks)
        text_embedding_pairs = list(zip(text_chunks, text_embeddings))
        db = PGVector.from_embeddings(text_embedding_pairs, embeddings,connection_string = connection_string)
        
        
        # Get the default Django database connection
        # CONNECTION_STRING = connections['default']
    #     CONNECTION_NAME = 'state_of_union_vectors'
        
    #     # Create PGVector from the provided documents
    #     db = PGVector.from_documents(
    #     embedding=embeddings,
    #     documents=text_chunks,  
    #     collection_name=CONNECTION_NAME,
    #     connection_string=CONNECTION_STRING
    # )
        
        response = user_input(user_question)  
        
        
        return render(request, 'index.html', {'response': response,'query': user_question })
    else:
        return render(request, 'index.html')

