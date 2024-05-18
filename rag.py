import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from pdf2image import convert_from_path
import fitz  # PyMuPDF

from langchain.text_splitter import CharacterTextSplitter
from sentence_transformers import SentenceTransformer

from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.llms import LlamaCpp
from langchain_core.documents import Document
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.prompts import PromptTemplate

import csv

import pickle

# Function to save vector store object
def save_vectorstore(vectorstore, filename):
    save_data = {
        "texts": vectorstore.texts,
        "embeddings": vectorstore.embeddings
    }
    with open(filename, 'wb') as file:
        pickle.dump(save_data, file)

# Function to load vector store object
def load_vectorstore(filename):
    with open(filename, 'rb') as file:
        vectorstore = pickle.load(file)
    return vectorstore

def get_pdf_data():
    directory = r"C:\Users\Venkatesh Dharmaraj\OneDrive\Desktop\DSPP\Project\lecture_slides"
    list_of_chunks = []
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if filename.endswith('.pdf'):
            # Create a folder for each PDF in the static directory
            pdf_folder = os.path.join("static", os.path.splitext(filename)[0])
            os.makedirs(pdf_folder, exist_ok=True)
            
            # Open the PDF file
            pdf_document = fitz.open(filepath)
            
            # Iterate through each page
            for i in range(len(pdf_document)):
                page = pdf_document.load_page(i)
                
                # Extract text from the page
                text = page.get_text()
                print(f"Page {i+1} Text:\n{text}")
                
                # Convert the page to an image
                pix = page.get_pixmap()
                image_path = os.path.join(pdf_folder, f"page_{i+1}.png")
                pix._writeIMG(image_path,format_='png',jpg_quality= None)
                print(f"Page {i+1} Image saved to: {image_path}")
                if len(text.split()) > 9:
            # Close the PDF document
                    list_of_chunks.append(Document(page_content=text, metadata= {"image_link":f'/static/{os.path.splitext(filename)[0]}/page_{i+1}.png', "video_link":'', "article_link":''}))
                print("Page", i+1, "saved as", image_path)
            pdf_document.close()

    return list_of_chunks

def get_text_from_pdf_or_txt():
    text = ""
    directory = r"C:\Users\Venkatesh Dharmaraj\OneDrive\Desktop\DSPP\Project\others"
    
    for filename in os.listdir(directory):
        print(filename)
        filepath = os.path.join(directory, filename)
        if filename.endswith('.pdf'):
            pdf_reader = PdfReader(filepath)
            for page in pdf_reader.pages:
                text += page.extract_text()
        elif filename.endswith('.txt'):
            with open(filepath, 'r') as file:
                text += file.read()
    return text

def read_csv_q_and_a():
    concatenated_values=[]
    csv_file= r"C:\Users\Venkatesh Dharmaraj\OneDrive\Desktop\DSPP\Project\q&a.csv"
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            # Assuming the CSV file has two columns
            if len(row) == 2:
                concatenated_values.append(row[0] + row[1])
    list_of_chunks = []
    for chunk in concatenated_values:
        list_of_chunks.append(Document(page_content=chunk, metadata= {}))
    return list_of_chunks

def get_piazza_posts():
    concatenated_values=[]
    csv_file= r"C:\Users\Venkatesh Dharmaraj\OneDrive\Desktop\DSPP\Project\piazza_posts.csv"
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            # Assuming the CSV file has two columns
            if len(row) == 2:
                concatenated_values.append(row[0] + row[1])
    list_of_chunks = []
    for chunk in concatenated_values:
        list_of_chunks.append(Document(page_content=chunk, metadata= {}))
    return list_of_chunks

def get_image_data():
    concatenated_values=[]
    csv_file= r"C:\Users\Venkatesh Dharmaraj\OneDrive\Desktop\DSPP\Project\image_data.csv"
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            # Assuming the CSV file has two columns
            if len(row) >= 2:
                concatenated_values.append(Document(page_content=row[0], metadata= {"image_link":row[1], "video_link":row[2], "timestamp": row[3], "article_link": row[4]}))
        # vectorstore = FAISS.load_local("faiss_index", OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY))
        # embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        # db = vectorstore.from_documents(concatenated_values, embeddings)
        # vectorstore.save_local("faiss_index")
    return concatenated_values


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=0,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    list_of_chunks = []
    for chunk in chunks:
        list_of_chunks.append(Document(page_content=chunk, metadata= {}))
    print(list_of_chunks)
    return list_of_chunks

def get_vectorstore(text_chunks, model_choice):
    if model_choice == "GPT3.5":
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    elif model_choice == "Llama":
         embeddings = HuggingFaceEmbeddings(model_name=r"C:\Users\Venkatesh Dharmaraj\Downloads\all-MiniLM-L6-v2")
    else:
        raise ValueError("Invalid model choice. Please select either 'GPT3.5' or 'Llama'.")

    # vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    vectorstore = FAISS.from_documents(text_chunks, embeddings)
    # save_vectorstore(vectorstore,"vector_db")
    print(vectorstore.index.ntotal)
    vectorstore.save_local("faiss_index")
    # print(vectorstore.docstore._dict)
    return vectorstore

def get_conversation_chain(model_choice):
    QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""Your task is to generate and add five  different versions of the given user question to retrieve relevant documents from a vector 
    database. By generating multiple perspectives on the user question, your goal is to help
    the user overcome some of the limitations of the distance-based similarity search. 
    Provide these alternative questions separated by newlines. Include the actual question too as one version without changing. Don't include any headings like 'Original Question' or 'Alternative Question'.
    Original question: {question}""",
)
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    vectorstore = FAISS.load_local("faiss_index", OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY))
    if model_choice == "GPT3.5":
        llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY)
    elif model_choice == "Llama":
        llm = HuggingFacePipeline.from_model_id(
        model_id=r"C:\Users\Venkatesh Dharmaraj\Downloads\vicuna-7b-v1.3",
        task="text-generation",
        model_kwargs={"temperature": 0.01, "max_length": 2000},
    )
    else:
        raise ValueError("Invalid model choice. Please select either 'GPT3.5' or 'Llama'.")
    # retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 7})
    
    retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(), llm=llm, prompt = QUERY_PROMPT)
    import logging

    logging.basicConfig()
    logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)


    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
    )
    return conversation_chain

# image_data= get_image_data()
# raw_text = get_text_from_pdf_or_txt()
# text_chunks_ = get_text_chunks(raw_text)
# text_chunks = get_pdf_data()
# csv_text = read_csv_q_and_a()
# piazza_posts = get_piazza_posts()
# model_choice = "GPT3.5"  
# print(text_chunks_)
# vectorstore = get_vectorstore(text_chunks_ + text_chunks + csv_text + piazza_posts + image_data, model_choice)
