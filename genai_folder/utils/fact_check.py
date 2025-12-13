import google.generativeai as genai
from tavily import TavilyClient
import json
from config import Config

gemini_api_key = Config.GEMINI_API_KEY
genai.configure(api_key=gemini_api_key)

model = genai.GenerativeModel("gemini-2.0-flash")
tavily_api_key = Config.TAVILY_API_KEY
tavily_client = TavilyClient(api_key=tavily_api_key)

def fact_check(claim):
    try:
        print(f"""Fact checking the claim: {claim}""")
        search_results = tavily_client.search(query=claim, 
                                               search_depth="advanced",
                                               max_results=5)
        if not search_results.get('results'):
            return {"claim": claim, "verdict": "No relevant information found to verify the claim.", "sources": []}

        evidence = " "
        source = []
        
        for result in enumerate(search_results.get('results',[])[:5]):
            title = result.get('title','Unknown Title-could not fetch')
            content = result.get('content','No content available')
            url = result.get('url','No URL available')
            
            evidence += f"Source { i+1}:Title: {title}\nContent: {content}\n\n"
            source.append({"title": title, "url": url})
            
        print("Generating verdict based on the evidence collected...")
        
        prompt = f"""
        You are a proffesional fact-checking algorithm. 
        Analyze this claim against the evidence:
        
        CLAIM : {claim}
        EVIDENCE: {evidence}
        Provide a fact-check analysis with:
        1. VERDICT: Choose ONE of these:
              - TRUE: Claim is factually correct based on the evidence.
              - FALSE: Claim is factually incorrect based on the evidence.
              - PARTIALLY TRUE: Claim is partially correct based on the evidence.
              - UNVERIFIABLE: Insufficient evidence to verify the claim.
              - OUTDATED : Claim was true but circumstances have changed.
              
        2. CONFIDENCE : High, Medium, Low
        
        3. EXPLANATION : Briefly explain in 2-3 lines the reasoning behind your verdict.
        
        4. KEY EVIDENCE : Summarize the most relevant pieces of evidence that influenced your verdict.
        
        Keep it concise and factual, do not respond to basic pleasantries. as fact checking is a serious task. only take up questions that you feel are genuine question about fact checking.
        """
        
        response = model.generate_content(prompt)
        analysis = response.text.strip()
        
        print("Fact-checking completed.")
        print('=================================')
        print(analysis)
        print('=================================')
        print("Sources:")
        for src in source:
            print(f"Title: {src['title']}, URL: {src['url']}")
    except Exception as e:
        print(f"Error during fact-checking: {e}")
        return {"claim": claim, "verdict": "Error during fact-checking.", "sources": []}
    
    
fc = fact_check("The Eiffel Tower is located in Berlin.")
print(fc)