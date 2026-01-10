import os
from typing import TypedDict, Annotated, List, Dict, Any
from datetime import datetime
import json
from dotenv import load_dotenv

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.tools import tool

from tavily import TavilyClient
from pinecone import Pinecone, ServerlessSpec

# Import prompts
from prompts import (
    SALES_ANALYST_SYSTEM,
    SALES_STRATEGIST_SYSTEM, 
    SALES_QUALIFIER_SYSTEM,
    ANALYSIS_PROMPT,
    PITCH_PROMPT,
    SCORING_PROMPT
)

# Load environment variables
load_dotenv()

# Langfuse setup
try:
    from langfuse import get_client
    from langfuse.decorators import observe, propagate_attributes
    
    if os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"):
        langfuse = get_client()
        if langfuse.auth_check():
            print("Langfuse tracing enabled")
            LANGFUSE_ENABLED = True
        else:
            print("Langfuse auth failed")
            LANGFUSE_ENABLED = False
    else:
        print("Langfuse keys not found")
        LANGFUSE_ENABLED = False
except ImportError:
    print("⚠ Langfuse not installed")
    LANGFUSE_ENABLED = False
    observe = lambda: lambda f: f  # No-op decorator
    
    # Create a no-op context manager
    from contextlib import nullcontext
    def propagate_attributes(**kwargs):
        return nullcontext()

GEMINI_API_KEY = os.getenv('GOOGLE_API_KEY') 
TAVILY_API_KEY = os.getenv('TAVILY_API_KEY')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')

#ini llm
if GEMINI_API_KEY:
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", 
                                 temperature=0.7,
                                 google_api_key=GEMINI_API_KEY)
else:
    print("Warning: GOOGLE_API_KEY not found, LLM will not work")
    llm = None

#Initiliz tavily
if TAVILY_API_KEY:
    tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
else:
    print("Warning: TAVILY_API_KEY not found, web search will not work")
    tavily_client = None

#ini Pinecone
if PINECONE_API_KEY:
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        INDEX_NAME = "sales-leads"
        if INDEX_NAME not in pc.list_indexes().names():
            pc.create_index(
                name=INDEX_NAME,
                dimension=768,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
            print(f"Created Pinecone index: {INDEX_NAME}")
    except Exception as e:
        print(f"Error creating Pinecone index: {e}")
else:
    print("Warning: PINECONE_API_KEY not found, vector database will not work")
    pc = None
    
    
class AgentState(TypedDict):
    company_name: str
    tavily_data: Dict[str, Any]
    crm_data: Dict[str, Any]
    analysis: str
    pitch: str
    action_items: List[str]
    confidence_score: float
    messages: List[Any]
    next_step: str
    
@tool
def search_company_info(company_name: str):
    """Search for company information using Tavily API."""
    try:
        if not tavily_client:
            return {"error": "Tavily client not initialized - missing API key"}
            
        overview_results = tavily_client.search(query=f"{company_name} company overview, headquarters, employees revenue", 
                                                search_depth="advanced",
                                                max_results=3)
        
        news_results = tavily_client.search(
            query=f"{company_name} news recent developements 2024 2025",
            search_depth="advanced",
            max_results=5
        )
        
        people_results = tavily_client.search(
            query=f"{company_name} CEO leadership team exceutives LinkedIn Board of Directors",
            search_depth="advanced",
            max_results=5
        )
        
        return {
            "overview": overview_results,
            "news": news_results,
            "people": people_results,
            "search_time":datetime.now().isoformat()
        }
    except Exception as e:
        print(f"Error searching company info: {e}")
        return {"error": str(e)}


@tool
def search_crm_data(company_name: str):
    """Search for CRM data related to the company."""
    try:
        mock_crm_data = {
                        "past_interactions": [
                            {
                            "date": "2024-11-15",
                            "contact": "John Doe",
                            "outcome": "Demo was schedules",
                            "notes": "Interested in enterprise feature"
                            }
                        ],
                        "similar_companies": [
                            {
                            "name": "Similar Tech Corp",
                            "deal_outcome": "Closed-Won",
                            "deal_size": "$50K ARR",
                            "key_factors": [
                                "Enterprise support",
                                "API Integrations"
                            ]
                            }
                        ],
                        "previous_pitches": [
                            {
                            "template": "enterprise_saas",
                            "success_rate": "0.55",
                            "best_subject_line": "Scaling companies with automated workflows as per clients requirements"
                            }
                        ]
                        }
        print(f"CRM data for {company_name}: {mock_crm_data}")
        return mock_crm_data
    except Exception as e:
        print(f"Error searching CRM data: {e}")
        return {"error": str(e)}

# ============================================================================
# AGENT NODES
# ============================================================================

def research_node(state: AgentState) -> AgentState:
    """Node 1: Gather information from Tavily and Pinecone"""
    print(f"\n🔍 Researching: {state['company_name']}")
    
    company_name = state["company_name"]
    
    # Search Tavily
    print("Searching web with Tavily...")
    tavily_data = search_company_info.invoke({"company_name": company_name})
    
    # Search CRM (Pinecone)
    print("  → Checking CRM data from Pinecone...")
    crm_data = search_crm_data.invoke({"company_name": company_name})
    
    state["tavily_data"] = tavily_data
    state["crm_data"] = crm_data
    state["next_step"] = "analyze"
    
    return state


@observe()
def analyze_node(state: AgentState) -> AgentState:
    """Node 2: Analyze gathered data with Gemini"""
    print("\nAnalyzing data with Gemini...")
    
    with propagate_attributes(
        user_id="sales_agent",
        session_id=f"research_{state['company_name'].lower()}",
        tags=["analysis", "gemini", "sales"],
        metadata={"company": state['company_name'], "step": "analysis"}
    ):
        # Prepare data summary for analysis
        tavily_summary = json.dumps(state["tavily_data"], indent=2)[:2000]  # Limit size
        crm_summary = json.dumps(state["crm_data"], indent=2)[:1000]
        
        analysis_prompt = ANALYSIS_PROMPT.format(
            company_name=state['company_name'],
            tavily_summary=tavily_summary,
            crm_summary=crm_summary
        )
        
        messages = [
            SystemMessage(content=SALES_ANALYST_SYSTEM),
            HumanMessage(content=analysis_prompt)
        ]
        
        response = llm.invoke(messages)
        
        if LANGFUSE_ENABLED:
            langfuse.update_current_trace(
                input=analysis_prompt,
                output=response.content,
                metadata={"model": "gemini-2.0-flash", "temperature": 0.7}
            )
        
        state["analysis"] = response.content
        state["next_step"] = "generate_pitch"
        
        print("Analysis complete")
        return state


@observe()
def generate_pitch_node(state: AgentState) -> AgentState:
    """Node 3: Generate personalized pitch with Gemini"""
    print("\nGenerating personalized pitch...")
    
    with propagate_attributes(
        user_id="sales_agent",
        session_id=f"research_{state['company_name'].lower()}",
        tags=["pitch", "gemini", "sales"],
        metadata={"company": state['company_name'], "step": "pitch_generation"}
    ):
        pitch_prompt = PITCH_PROMPT.format(
            company_name=state['company_name'],
            analysis=state['analysis']
        )
        
        messages = [
            SystemMessage(content=SALES_STRATEGIST_SYSTEM),
            HumanMessage(content=pitch_prompt)
        ]
        
        response = llm.invoke(messages)
        
        if LANGFUSE_ENABLED:
            langfuse.update_current_trace(
                input=pitch_prompt,
                output=response.content,
                metadata={"model": "gemini-2.0-flash", "temperature": 0.7}
            )
        
        state["pitch"] = response.content
        state["next_step"] = "score_lead"
        
        print("Pitch generated")
        return state


@observe()
def score_lead_node(state: AgentState) -> AgentState:
    """Node 4: Score the lead and generate action items"""
    print("\nScoring lead quality...")
    
    with propagate_attributes(
        user_id="sales_agent",
        session_id=f"research_{state['company_name'].lower()}",
        tags=["scoring", "gemini", "sales"],
        metadata={"company": state['company_name'], "step": "lead_scoring"}
    ):
        scoring_prompt = SCORING_PROMPT.format(
            company_name=state['company_name'],
            analysis=state['analysis'][:500]
        )
        
        messages = [
            SystemMessage(content=SALES_QUALIFIER_SYSTEM),
            HumanMessage(content=scoring_prompt)
        ]
        
        response = llm.invoke(messages)
        
        if LANGFUSE_ENABLED:
            langfuse.update_current_trace(
                input=scoring_prompt,
                output=response.content,
                metadata={"model": "gemini-2.0-flash", "temperature": 0.7}
            )
        
        # Parse score (simple extraction)
        content = response.content
        try:
            score_line = [line for line in content.split('\n') if 'SCORE:' in line][0]
            score = float(score_line.split('/')[0].split(':')[1].strip())
        except:
            score = 7.0  # Default
        
        # Extract action items
        try:
            action_section = content.split('ACTION ITEMS:')[1]
            action_items = [line.strip() for line in action_section.split('\n') if line.strip() and line.strip()[0].isdigit()]
        except:
            action_items = ["Follow up with research", "Prepare personalized demo", "Connect on LinkedIn"]
        
        state["confidence_score"] = score
        state["action_items"] = action_items
        state["next_step"] = "save_results"
        
        print(f"Lead scored: {score}/10")
        return state


def save_results_node(state: AgentState) -> AgentState:
    """Node 5: Save results to Pinecone and prepare final output"""
    print("\n💾 Saving results...")
    
    # In production, save to Pinecone here
    # For demo, just print confirmation
    
    result_data = {
        "company": state["company_name"],
        "research_date": datetime.now().isoformat(),
        "score": state["confidence_score"],
        "analysis": state["analysis"][:500],
        "pitch": state["pitch"][:500]
    }
    
    print(f"  ✓ Results saved for {state['company_name']}")
    # In production: pinecone_index.upsert(vectors=...)
    
    state["next_step"] = "end"
    return state


def should_continue(state: AgentState) -> str:
    """Router function to determine next step"""
    return state.get("next_step", "end")


# ============================================================================
# BUILD LANGGRAPH
# ============================================================================

def create_agent_graph():
    """Create the LangGraph workflow"""
    
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("research", research_node)
    workflow.add_node("analyze", analyze_node)
    workflow.add_node("generate_pitch", generate_pitch_node)
    workflow.add_node("score_lead", score_lead_node)
    workflow.add_node("save_results", save_results_node)
    
    # Add edges
    workflow.set_entry_point("research")
    
    workflow.add_edge("research", "analyze")
    workflow.add_edge("analyze", "generate_pitch")
    workflow.add_edge("generate_pitch", "score_lead")
    workflow.add_edge("score_lead", "save_results")
    workflow.add_edge("save_results", END)
    
    return workflow.compile()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

@observe()
def research_company(company_name: str) -> Dict[str, Any]:
    """Main function to research a company"""
    
    with propagate_attributes(
        user_id="sales_agent",
        session_id=f"research_{company_name.lower()}",
        tags=["sales_research", "workflow"],
        metadata={"company": company_name, "workflow": "sales_lead_research"}
    ):
        print("=" * 70)
        print(f"🚀 SALES LEAD RESEARCHER AGENT")
        print("=" * 70)
        
        # Initialize state
        initial_state = {
            "company_name": company_name,
            "tavily_data": {},
            "crm_data": {},
            "analysis": "",
            "pitch": "",
            "action_items": [],
            "confidence_score": 0.0,
            "messages": [],
            "next_step": "research"
        }
        
        # Create and run agent
        agent = create_agent_graph()
        final_state = agent.invoke(initial_state)
        
        if LANGFUSE_ENABLED:
            langfuse.update_current_trace(
                input={"company_name": company_name},
                output={
                    "analysis_length": len(final_state["analysis"]),
                    "pitch_length": len(final_state["pitch"]),
                    "confidence_score": final_state["confidence_score"],
                    "action_items_count": len(final_state["action_items"])
                },
                metadata={"workflow_completed": True}
            )
        
        return final_state


def format_output(result: Dict[str, Any]) -> str:
    """Format the final output nicely"""
    
    output = f"""
{'=' * 70}
📊 RESEARCH REPORT: {result['company_name']}
{'=' * 70}

🔍 ANALYSIS:
{result['analysis']}

{'─' * 70}

✍️  PERSONALIZED PITCH:
{result['pitch']}

{'─' * 70}

🎯 LEAD SCORE: {result['confidence_score']}/10

📋 ACTION ITEMS:
"""
    
    for i, action in enumerate(result['action_items'], 1):
        output += f"{i}. {action}\n"
    
    output += f"\n{'=' * 70}\n"
    
    return output


# ============================================================================
# RUN THE AGENT
# ============================================================================

if __name__ == "__main__":
    
    # Example 1: Research a company
    company_to_research = "Stripe"  # Change this to any company
    
    try:
        result = research_company(company_to_research)
        print(format_output(result))
        
        # Optionally save to file
        with open(f"research_{company_to_research.lower().replace(' ', '_')}.txt", "w") as f:
            f.write(format_output(result))
        print(f"✓ Report saved to research_{company_to_research.lower().replace(' ', '_')}.txt")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure you've set your API keys at the top of the file!")
    
    # Example 2: Research multiple companies
    # companies = ["Stripe", "Shopify", "Square"]
    # for company in companies:
    #     result = research_company(company)
    #     print(format_output(result))
    #     print("\n" + "="*70 + "\n")
