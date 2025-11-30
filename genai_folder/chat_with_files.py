import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
import pandas as pd
from utils.config import Config
import warnings
warnings.filterwarnings("ignore")

gemini_api_key = Config.GEMINI_API_KEY

genai.configure(api_key=gemini_api_key)

#initilaize my llm 
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", 
                                    google_api_key=gemini_api_key,
                                    temperature=0.7,
                                    max_output_tokens=512
                                    )

def load_from_google_sheet(sheet_url):
    sheet_id = sheet_url.split('/d/')[1].split('/')[0]
    csv_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
    df = pd.read_csv(csv_url)
    df.columns = df.columns.str.strip()
    duplicates = (df.duplicated()).sum()
    print(f"Found {duplicates} duplicate rows")
    #duplicate rows
    if duplicates > 0:
        df.drop_duplicates(inplace=True)
        
    print(f"Found Columns: {df.columns.tolist()}\n")
    
    chunks=[]
    for idx, row in df.iterrows():
        if row.isna().all():
            continue
        #biuild chunk qith all non empty columns
        chunk_parts = []
        
        for col in df.columns:
            value = row[col]
            if pd.notna(value) and str(value).strip() != '':
                chunk_parts.append(f"{col}: {value}")
                
        if not chunk_parts:
            continue
        
        chunk_text = "\n".join(chunk_parts)
        
        #metada -> better retreive
        metadata = {
            "row_number":idx,
            "product_name":str(row.get('Product Name (Clean)','')),
            "brand":str(row.get("Brand",'')),
            "category":str(row.get("Category", '')),
            "pack_size":str(row.get("Pack Size Options",''))
            }
        
        chunks.append({
            "text":chunk_text,
            "metadata":metadata
        })
    return chunks

print(f"="*60)
print("Loading prodiuct data from google sheet........!!")
print(f"="*60)      

sheet_url = Config.SHEET_URL

try:
    chunks = load_from_google_sheet(sheet_url)
    print(f"Created {len(chunks)} Products Chunks\n")
except Exception as e:
    print(f"Error loading data from Google Sheet: {e}")


docs = [Document(page_content=chunk["text"],metadata=chunk["metadata"]) for chunk in chunks]

print("Creating vector embeddings.....")
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
vector_store = FAISS.from_documents(docs, embedding_model)

def search_documents(query, k=3):
    results = vector_store.similarity_search(query, k=k)
    
    combined_results = []
    for i, res in enumerate(results):
        product_info = f"==== Product {i+1} ====\n"
        product_info += res.page_content
        combined_results.append(product_info)
    return f"Found {len(results)} relevant products:\n\n" + "\n\n".join(combined_results)

search_tools = Tool(
    name="ProductSearch",
    func=search_documents,
    description="Use this tool to search for relevant products based on the query."
)
    
agent = initialize_agent(
    tools=[search_tools],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    max_iterations=3,
    handle_parsing_errors=True
)   

instruction = """
You are a helpful asian grocery store assistant, very experienced , very knowledable, analytical and
you do work with detail oriented, proffesional.

1.when a user asks for a query, use the ProductSearch tool to find relevant products and then provide a
concise answer to the user.
2.Do not accept foul language, do not accept hateful and toxic language.
3.Do not answer any queries not related to companies information, for such naswers ask user to ask wirstions related to company ot data in vectordb
"""

user_query = "what brand do i use for selling besan cookies"
query = f"{instruction}\n\n{user_query}"
response= agent.run(query)
# print(f"USER: {query}")
# print(f"AI: {response}")

ls =  load_from_google_sheet(sheet_url)
print(ls)