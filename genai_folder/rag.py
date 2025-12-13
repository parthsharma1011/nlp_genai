
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from langchain.agents import initialize_agent, Tool, AgentType
import google.generativeai as genai
import os
import warnings
warnings.filterwarnings("ignore")

gemini_api_key = os.getenv('GEMINI_API_KEY', 'your_api_key_here')
genai.configure(api_key=gemini_api_key)

#initilaize my llm 
gemini_llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", 
                                    google_api_key=gemini_api_key,
                                    temperature=0.7,
                                    max_output_tokens=512
                                    )

#pdf, databses, csv, text, etc
documents = [
  "ABC.Informatic was founded in 2010 to deliver innovative IT solutions.",
  "The company specializes in cloud computing, cybersecurity, and enterprise software.",
  "ABC.Informatic has offices in Berlin, Munich, and Hamburg with over 500 employees.",
  "Our mission is to simplify digital transformation for businesses worldwide.",
  "ABC.Informatic invests heavily in research and development to stay ahead in technology."
]

#python list 
docs = [Document(page_content=doc) for doc in documents]

# print('=='*40)
# print("Original Structure\n")
# print(documents)
# print('=='*40)
# print('=='*40)
# print("New Structure\n")
# print(docs)
# print('=='*40)

#embeddings "i love india " -> text
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(docs, embeddings)


def search_documents(query):
    results = vectorstore.similarity_search(query, k=2) #top 2 similar doc
    combined = "\n".join([f"- {res.page_content}" for res in results])
    return f"Relevant Documents:\n{combined}"

search_tool = Tool(
    name="VectorStoreLookup",
    func=search_documents,
    description="Use this tool to search for relevant documents based on the query."
)

agent = initialize_agent(
    tools = [search_tool],
    llm = gemini_llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)

instructions = ("You are a reasearch assistant and an agent"
                "when asked a question, think step-by-step"
                "if you need more info, use the VectorStoreLookup tool by writing your though process, then call the tool"
                "after getting the info, summarize it and provide a final concise answer to the user."
                "Do not acceot foul language, do not accept hateful and toxic language"
                "do not answer any queries not related to companies information, for such naswers ask user to ask wirstions related to company ot data in vectodb")

query = f"{instructions}\n\nwhat does ABC.Informatic company do?"
response= agent.run(query)
print(f"USER: {query}")
print(f"AI: {response}")

#logic -> llm -> company -> gaurdrails.

# query = f"{instructions}\n\n:"
# response= agent.run(query)
# print(f"USER: {query}")
# print(f"AI: {response}")

#Namaste Folks, Lets start in 2-4 minutes