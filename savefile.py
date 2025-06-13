from langchain_core.output_parsers.string import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeEmbeddings, PineconeVectorStore
from langchain_community.document_loaders.pdf import PyPDFLoader
import sqlite3
from flask import Flask, request
from werkzeug.utils import secure_filename
from pinecone import Pinecone
from dotenv import load_dotenv
import os

#keys
load_dotenv()
PINECONE_API = os.getenv("PINECONE_API_KEY")
GEMINI_API = os.getenv("GEMINI_KEY")

#initialisation

llm = ChatGoogleGenerativeAI(model='models/gemini-2.0-flash-lite', api_key=GEMINI_API)
pc = Pinecone(api_key="YOUR_API_KEY")

UPLOAD_FOLDER = r'C:\Users\smile\Desktop\Python\RAG\Files'
ALLOWED_EXTENSIONS = {'txt', 'pdf'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


embeddings = PineconeEmbeddings(
    model="multilingual-e5-large",
    pinecone_api_key=os.environ["PINECONE_API_KEY"]
)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/upload", methods=['POST'])
def post_file():
    file = request.files['file']
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    single_doc = PyPDFLoader(file_path=path, mode='single')
    docs = single_doc.load()
    txt_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size = 300,
        chunk_overlap = 50)
    split_docs = txt_splitter.split_documents(docs)
    

    vectorstore = PineconeVectorStore(
    index_name="new-app",
    embedding=embeddings,
    pinecone_api_key=os.environ["PINECONE_API_KEY"]
    )

    vectorstore.add_documents(split_docs)

    return {"result" : "success"}

@app.route("/save_to_db", methods = ['POST'])
def save_in_db():
    #sql initialization
    conn = sqlite3.connect("document.db")
    cursor = conn.cursor()
    cursor.execute('CREATE TABLE IF NOT EXISTS documents (document_name TEXT, description TEXT)')
    conn.commit()

    file = request.files['file']
    path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    loader = PyPDFLoader(path)
    doc = loader.load()

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size = 600, chunk_overlap = 50)
    splits = text_splitter.split_documents(doc)

    summaries = []
    prompt1 = "create one line description of what topic you think the the topic belongs to {content}"
    prop = PromptTemplate.from_template(prompt1)
    parser = StrOutputParser()
    for split in splits:
        chain = prop | llm | parser
        summaries.append(chain.invoke({"content" : split}))
    prompt2 = "you are a great summarizer for documents given the list of content in a file create one line description of everything in the file the list of content is {list_of_content} Output the result as a plain string. Do not use Markdown formatting, bullet points, or special characters "
    prop2 = PromptTemplate.from_template(prompt2)
    chain = prop2 | llm | parser
    summary = chain.invoke({"list_of_content" : summaries})
    cursor.execute("INSERT INTO DOCUMENTS (document_name, description) VALUES(?,?)",(file.filename, summary))
    conn.commit()
    return {"result" : "success"}

if __name__ == "__main__":
    app.run(port=5000, debug= True)