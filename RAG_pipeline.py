from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END, START
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_pinecone import PineconeEmbeddings, PineconeVectorStore
from langchain_core.output_parsers import StrOutputParser
import os
from typing import Optional
from dotenv import load_dotenv
import sqlite3
from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

SERPER_API_KEY = os.getenv("SERPER_API_KEY")
PINECONE_API = os.getenv("PINECONE_API_KEY")
GEMINI_API = os.getenv("GEMINI_KEY")

llm = ChatGoogleGenerativeAI(model='models/gemini-2.0-flash-lite', api_key=GEMINI_API)

parser = StrOutputParser()

embedding = PineconeEmbeddings(
    model="multilingual-e5-large",
    pinecone_api_key=PINECONE_API
)
vector_store = PineconeVectorStore(index_name="new-app", embedding=embedding,
    pinecone_api_key=PINECONE_API)

retriever = vector_store.as_retriever(
    search_kwargs={"k": 3},
)

class Router(BaseModel):
    """"""" ROUTES TO EITHER WEB OR RAG """""""

    rag : str = Field(description=f"Are content of document relevant, 'yes' or 'no'")



prop =  """You are a router that decides whether a user's question should be answered using RAG.
If any topic in the list of summaries is relevant to the user question, output 'yes' in the 'rag' field.
Otherwise, output 'no'.\n\nQuery: {query}\n\nTopics: {summaries}"""
prompt = PromptTemplate.from_template(prop)

route_parser = llm.with_structured_output(Router)
router_chain = prompt | route_parser
# result = router_chain.invoke({"query" : "Who is Worlds Strongest Man", "summaries" : summaries})
# print(result.rag)

class binary_score(BaseModel):

    result : str = Field("Is the document relevant, yes or no")

grader_prop = "is there any passage relevant to the question  output 'yes' in the 'binary_score' field. Otherwise, output 'no'. \n\n question : {question} \n\n document : {document} "
grader_prompt = PromptTemplate.from_template(grader_prop)

binary_score_router = llm.with_structured_output(binary_score)
grade_chain = grader_prompt | binary_score_router 

gen_prop = "You are a agent with excellent writing skills given the question and context give a satisifying and clear response, respond in plain text, no highlight \n Question :{question} \n Context : {context} "

gen_prompt = PromptTemplate.from_template(gen_prop)

gen_chain = gen_prompt | llm | parser
class hallucinate_test(BaseModel):

    result : str = Field("Is this fact based in truth, yes or no")

hallucinate_prop = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n
Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts. \n
Generated response : {response}"""
hallucinate_prompt = PromptTemplate.from_template(hallucinate_prop)
hallucinate_router = llm.with_structured_output(hallucinate_test)
hallucinate_chain = hallucinate_prompt | hallucinate_router

class State(BaseModel):
    query : str = Field("Question from user")
    context : Optional[str] = None
    response : Optional[str] = None

def router(state : State):
    conn = sqlite3.connect("document.db")
    cursor = conn.cursor()
    cursor.execute("SELECT description FROM documents")
    output = cursor.fetchall()
    summaries = []
    for entry in output:
        summaries.append(entry)
    result = router_chain.invoke({"query" : state.query, "summaries" : summaries})
    if result.rag == "yes":
        return "RAG"
    else :
        return "WEB"

def retrieval(state : State) -> State:
    print("------------------RETRIEVING----------------------")
    docs = retriever.invoke(state.query)
    list_of_content = [doc.page_content for doc in docs]
    context = "\n".join(list_of_content)
    return State(
        query = state.query,
        context = context,
        response = state.response
    )

def websearch(state: State) -> State:
    websearch = GoogleSerperAPIWrapper(serper_api_key=SERPER_API_KEY)
    print("------------------WEB_SEARCH----------------------")
    web_result = websearch.run(query=state.query)
    return State(
        query = state.query,
        context = web_result,
        response = state.response
    ) 

def doc_grader(state: State):
    print("------------------DOC_GRADER----------------------")
    result = grade_chain.invoke({"question" : state.query, "document" : state.context})
    if(result.result == "no"):
        return "WEB"
    else:
        return "GEN" 
    
def generate(state: State) -> State:
    print("------------------GENERATOR----------------------")
    result = gen_chain.invoke({"question" : state.query, "context" : state.context})
    return State(
        query = state.query,
        context = state.context,
        response = result
    )

def chk_hal(state: State):
    print("------------------HALLUCINATION----------------------")
    result = hallucinate_chain.invoke({"response" : state.response})
    if result.result == "yes":
        return "END"
    else: 
        return "GEN"
    
workflow = StateGraph(State)
workflow.add_node('router', router)
workflow.add_node('rag', retrieval)
workflow.add_node('web_search', websearch)
workflow.add_node('eval', doc_grader)
workflow.add_node('generator', generate)
workflow.add_node('hallucinate', chk_hal)

workflow.add_conditional_edges(START,
        router,
        {
            'RAG':'rag',
            'WEB' : 'web_search'
        })
workflow.add_edge('web_search', 'generator')
workflow.add_conditional_edges('rag',
        doc_grader,
        {
            "WEB" : 'web_search',
            "GEN" : "generator"
        })
workflow.add_conditional_edges('generator',
        chk_hal,
        {
            "END" : END,
            "GEN" : "generator"
        })

app = workflow.compile()
endpoint = FastAPI()
class QueryInput(BaseModel):
        query: str
    
@endpoint.post("/")
def submit_query(query: str = Form(...)):
    state_input = {"query": query}
    result = app.invoke(state_input)
    return {"response": result["response"]}

endpoint.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(endpoint, port=8000)




 
