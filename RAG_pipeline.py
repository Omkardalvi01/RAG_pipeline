from pydantic import BaseModel
from langchain_community.vectorstores import Chroma
from langgraph.graph import StateGraph, END
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import uuid
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_community.utilities import GoogleSerperAPIWrapper

class State(BaseModel):
    query : str
    document : str
    reponse : str



 
