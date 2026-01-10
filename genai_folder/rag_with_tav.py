from langchain_google_genai import ChatGoogleGenerativeAI
from tavily import TavilyClient
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from langchain.agents import initialize_agent, Tool, AgentType
import google.generativeai as genai
from utils.config import Config

import warnings
warnings.filterwarnings("ignore")

gemini_api_key = Config.GEMINI_API_KEY
genai.configure(api_key=gemini_api_key)

tavily_api_key = Config.TAVILY_API_KEY
tavily_client = TavilyClient(api_key=tavily_api_key)

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", 
                                    google_api_key=gemini_api_key,
                                    temperature=0.7,
                                    max_output_tokens=512)

documents = [
    "Ritu Kohli works as a data scientist at TechCorp, focusing on machine learning and AI projects.",
    "Soham Chaterji is also working as a data scientist at TechCorp, specializing in data analysis and visualization. but he has experience in machine learning as well.",
    "Ritu has background in marketing and sales and she handles the workload of many marketing executives.",
]

docs = [Document(page_content=doc) for doc in documents]

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(docs, embeddings)

#Tool -> TAG 
def search_documents(query):
    results = vectorstore.similarity_search(query, k=2)
    if not results:
        return "No relevant documents found."
    combined = "\n".join([f"- {res.page_content}" for res in results])
    if len(combined.strip()) < 20:
        return "No relevant documents found."
    return f"Relevant Documents:\n{combined}"

def search_tavily(query):
    try:
        print(f"Fallback triggered. Searching Tavily for: {query}")
        search_results = tavily_client.search(query=query, 
                                              search_depth="advanced", 
                                              max_results=3)
        
        if not search_results.get('results'):
            return "No relevant information found in Tavily."
        
        web_info = []
        for i, result in enumerate(search_results.get('results', [])[:3],1):
            title = result.get('title', 'Unknown Title - could not fetch')
            content = result.get('content', 'No content available')
            url = result.get('url', 'No URL available')
            web_info.append(f"Source {i}:\nTitle: {title}\nContent: {content}\nURL: {url}\n")
        
        return "Information from web is as follows:\n" + "\n".join(web_info)
    
    except Exception as e:
        print(f"Error during Tavily search: {e}")
        return "An error occurred while searching for information."
    
    
def search_with_fallback(query):
    doc_results = search_documents(query)
    
    if "No relevant documents found." in doc_results:
        print("No relevant documents found in vector store. Falling back to Tavily search.")
        return search_tavily(query)
    
    if len(doc_results) < 50:
        print("Insufficient information from vector store. Falling back to Tavily search.")
        web_results = search_tavily(query)
        return f"{doc_results}\n\nAdditional Information from web:\n{web_results}"
        
    return doc_results

#create tool
search_tool = Tool(
    name = "DocumentSearch",
    description="Use this tool to search for relevant documents or web information based on the query.",
    func = search_with_fallback
)

tavily_tool = Tool(
    name = "WebSearch",
    description="Use this tool to search the web for relevant information based on the query.",
    func = search_tavily
)

agent = initialize_agent(
    tools = [search_tool, tavily_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True
)

instructions = """
"You are a research assistant and an agent that helps users find information from internal documents and the web. 
Use the DocumentSearch tool to look for relevant documents first. 
If the documents do not provide sufficient information, 
use the WebSearch tool to gather additional information from the web.
When asked a questionm think step-by-step.
1. ALWAYS try DocumentSearch FIRST to check internal knowledge base.
2. ONly use Websearch if DocumentSearch returns insufficient or no information.
3. After gathering information, provide a clear and concise answer to the user.
4. If Information comes from the web, mention that it is from external sources.
5. Do not accept foul language, hateful or toxic language.
6. Do not answer any queries not related to company information; for such answers, ask the
"""

query_1 = f"{instructions}\n\nWhat does Ritu Kohli do in TechCorp?"
response_1 = agent.run(query_1)
print(f"USER: {query_1}")
print(f"AI: {response_1}")