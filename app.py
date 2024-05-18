from flask import Flask, request, jsonify, send_from_directory
import os
from werkzeug.utils import secure_filename
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.retrievers.multi_query import MultiQueryRetriever


from rag import get_text_from_pdf_or_txt, get_text_chunks, get_vectorstore, get_conversation_chain

app = Flask(__name__, static_url_path='/static')
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
vectorstore = FAISS.load_local("faiss_index", OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY))
# llm = ChatOpenAI(temperature=0)
# retriever_from_llm = MultiQueryRetriever.from_llm(
#     retriever=vectordb.as_retriever(), llm=llm
# )

processed_data = {
    'text_chunks': [],
    'vectorstore': None,
    'conversation_chain': None,
}

ALLOWED_EXTENSIONS = {'pdf', 'txt'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
#route to pdfs
@app.route('/upload', methods=['POST'])
def upload():
    print("yes")
    file = request.files['file']
    docs_paths = []
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        docs_paths.append(filepath)

    # if docs_paths:
    #     raw_text = get_text_from_pdf_or_txt(docs_paths)
    #     text_chunks = get_text_chunks(raw_text)
    #     model_choice = "GPT3.5"  
    #     vectorstore = get_vectorstore(text_chunks, model_choice)
    #     conversation_chain = get_conversation_chain(vectorstore, model_choice)
    #     global processed_data
    #     # Store processed data for later use
    #     # processed_data['text_chunks'] = text_chunks
    #     # processed_data['vectorstore'] = vectorstore
    #     processed_data['conversation_chain'] = conversation_chain
    
    return jsonify({"message": "Files successfully uploaded and processed"})

@app.route('/process-question', methods=['POST'])
def process_question():
    data = request.get_json()
    question = data.get('question', '')
    # llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY)

    # Retrieve the conversation chain from stored data
    global processed_data
    conversation_chain = processed_data['conversation_chain']

    # Process the question
    try:
        response = conversation_chain({'question': question}) 
        contexts = vectorstore.similarity_search(question, k=4)
        print(len(contexts))
        images,videos,timestamps,articles=[],[],[],[]
        for i in contexts:
            if i.metadata:
                if i.metadata['image_link']:
                    images.append(i.metadata['image_link'])
                if i.metadata['video_link']:
                    videos.append(i.metadata['video_link'])
                    # timestamps.append(i.metadata['timestamp'])
                if i.metadata['article_link']:
                    articles.append(i.metadata['article_link'])
        return jsonify({"response": response['answer'],"images":images, "videos": videos, "articles":articles})
    except:
        return jsonify({"response": "Apologies, but I currently lack sufficient information. Rest assured, we're actively addressing this issue and anticipate providing a response shortly."})

# route to html
@app.route('/')
def index():
    # raw_text = get_text_from_pdf_or_txt()
    # text_chunks = get_text_chunks(raw_text)
    model_choice = "GPT3.5"  
    # vectorstore = get_vectorstore(text_chunks, model_choice)
    conversation_chain = get_conversation_chain(model_choice)
    global processed_data
    # # Store processed data for later use
    # processed_data['text_chunks'] = text_chunks
    # processed_data['vectorstore'] = vectorstore
    processed_data['conversation_chain'] = conversation_chain
    return send_from_directory('.', 'p2.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
