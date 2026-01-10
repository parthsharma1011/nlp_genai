# Sales Lead Research Agent - Code Structure & Architecture

##  Overview
An AI-powered sales agent that automates company research, analysis, and personalized pitch generation using LangGraph workflow orchestration.

##  Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    SALES LEAD RESEARCH AGENT                    │
├─────────────────────────────────────────────────────────────────┤
│  Input: Company Name → Output: Research Report + Action Items   │
└─────────────────────────────────────────────────────────────────┘

┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   Research   │───▶│   Analysis   │───▶│    Pitch     │───▶│   Scoring    │───▶│    Save      │
│     Node     │    │     Node     │    │ Generation   │    │     Node     │    │   Results    │
│              │    │              │    │     Node     │    │              │    │     Node     │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
```

##  File Structure

```
sales_agent/
├── agent.py           # Main agent logic & workflow
├── prompts.py         # External prompt templates
├── .env              # API keys & configuration
├── requirements.txt   # Python dependencies
└── README.md         # This documentation
```

##  Core Components

### 1. **Dependencies & Imports**
```python
# Core Framework
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI

# External APIs
from tavily import TavilyClient          # Web search
from pinecone import Pinecone           # Vector database
from langfuse import get_client         # Observability

# Prompts (External)
from prompts import (
    SALES_ANALYST_SYSTEM,
    ANALYSIS_PROMPT,
    PITCH_PROMPT,
    SCORING_PROMPT
)
```

### 2. **State Management**
```python
class AgentState(TypedDict):
    company_name: str           # Target company
    tavily_data: Dict          # Web research results
    crm_data: Dict             # CRM/historical data
    analysis: str              # AI-generated analysis
    pitch: str                 # Personalized pitch
    action_items: List[str]    # Next steps
    confidence_score: float    # Lead quality (0-10)
    next_step: str            # Workflow control
```

### 3. **API Integrations**

#### **Google Gemini (LLM)**
```python
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.7,
    google_api_key=GEMINI_API_KEY
)
```

#### **Tavily (Web Search)**
```python
@tool
def search_company_info(company_name: str):
    # Searches: company overview, recent news, leadership
    return {
        "overview": overview_results,
        "news": news_results, 
        "people": people_results
    }
```

#### **Pinecone (Vector DB)**
```python
# Stores and retrieves CRM data, similar companies, past pitches
pc = Pinecone(api_key=PINECONE_API_KEY)
INDEX_NAME = "sales-leads"
```

#### **Langfuse (Observability)**
```python
@observe()  # Traces function execution
def analyze_node(state):
    with propagate_attributes(
        session_id=f"research_{company_name}",
        tags=["analysis", "gemini", "sales"]
    ):
        # Function logic with automatic tracing
```

##  Workflow Nodes

### **Node 1: Research** 
```python
def research_node(state: AgentState) -> AgentState:
    # Gathers data from multiple sources
    tavily_data = search_company_info.invoke({"company_name": company_name})
    crm_data = search_crm_data.invoke({"company_name": company_name})
    
    state["tavily_data"] = tavily_data
    state["crm_data"] = crm_data
    state["next_step"] = "analyze"
    return state
```

**Inputs:** Company name  
**Outputs:** Web research data, CRM data  
**APIs Used:** Tavily, Pinecone  

### **Node 2: Analysis** 
```python
@observe()
def analyze_node(state: AgentState) -> AgentState:
    analysis_prompt = ANALYSIS_PROMPT.format(
        company_name=state['company_name'],
        tavily_summary=tavily_summary,
        crm_summary=crm_summary
    )
    
    response = llm.invoke([
        SystemMessage(content=SALES_ANALYST_SYSTEM),
        HumanMessage(content=analysis_prompt)
    ])
    
    state["analysis"] = response.content
    return state
```

**Inputs:** Raw research data  
**Outputs:** Structured company analysis  
**APIs Used:** Google Gemini  
**Tracing:** Full Langfuse observability  

### **Node 3: Pitch Generation** 
```python
@observe()
def generate_pitch_node(state: AgentState) -> AgentState:
    pitch_prompt = PITCH_PROMPT.format(
        company_name=state['company_name'],
        analysis=state['analysis']
    )
    
    response = llm.invoke([
        SystemMessage(content=SALES_STRATEGIST_SYSTEM),
        HumanMessage(content=pitch_prompt)
    ])
    
    state["pitch"] = response.content
    return state
```

**Inputs:** Company analysis  
**Outputs:** Personalized sales pitch (email subjects, body, talking points)  
**APIs Used:** Google Gemini  

### **Node 4: Lead Scoring** 
```python
@observe()
def score_lead_node(state: AgentState) -> AgentState:
    # Scores lead quality 0-10 and extracts action items
    score = extract_score(response.content)  # Parse "SCORE: X/10"
    action_items = extract_actions(response.content)
    
    state["confidence_score"] = score
    state["action_items"] = action_items
    return state
```

**Inputs:** Company analysis  
**Outputs:** Lead score (0-10), specific action items  
**APIs Used:** Google Gemini  

### **Node 5: Save Results** 
```python
def save_results_node(state: AgentState) -> AgentState:
    result_data = {
        "company": state["company_name"],
        "research_date": datetime.now().isoformat(),
        "score": state["confidence_score"],
        "analysis": state["analysis"][:500],
        "pitch": state["pitch"][:500]
    }
    # In production: save to Pinecone vector database
    return state
```

##  Prompt Engineering

### **External Prompt Management**
All prompts are stored in `prompts.py` for easy maintenance:

```python
# prompts.py
ANALYSIS_PROMPT = """
You are a sales intelligence analyst. Analyze the following data about {company_name}:

WEB RESEARCH DATA:
{tavily_summary}

CRM DATA:
{crm_summary}

Provide a concise analysis covering:
1. Company overview (size, industry, recent developments)
2. Key decision makers and contacts
3. Recent news that indicates potential needs or pain points
4. Opportunities for our sales approach
5. Any red flags or concerns
"""
```

### **Dynamic Prompt Injection**
```python
analysis_prompt = ANALYSIS_PROMPT.format(
    company_name=state['company_name'],
    tavily_summary=tavily_summary,
    crm_summary=crm_summary
)
```

##  Observability & Tracing

### **Langfuse Integration**
```python
# Modern decorator-based tracing
@observe()
def analyze_node(state: AgentState):
    with propagate_attributes(
        user_id="sales_agent",
        session_id=f"research_{company_name.lower()}",
        tags=["analysis", "gemini", "sales"],
        metadata={"company": company_name, "step": "analysis"}
    ):
        # Automatic input/output logging
        if LANGFUSE_ENABLED:
            langfuse.update_current_trace(
                input=analysis_prompt,
                output=response.content,
                metadata={"model": "gemini-2.0-flash"}
            )
```

**Tracks:**
- All LLM calls with input/output
- Token usage and costs
- Session-based workflow tracing
- Performance metrics
- Error handling

##  Configuration

### **Environment Variables**
```bash
# .env file
GOOGLE_API_KEY=your_gemini_api_key
TAVILY_API_KEY=your_tavily_api_key  
PINECONE_API_KEY=your_pinecone_api_key
LANGFUSE_PUBLIC_KEY=pk-lf-your-key
LANGFUSE_SECRET_KEY=sk-lf-your-key
LANGFUSE_HOST=https://cloud.langfuse.com
```

### **Graceful Degradation**
```python
# Each API client has fallback handling
if GEMINI_API_KEY:
    llm = ChatGoogleGenerativeAI(...)
else:
    print("Warning: GOOGLE_API_KEY not found")
    llm = None
```

##  Execution Flow

### **Main Entry Point**
```python
@observe()
def research_company(company_name: str) -> Dict[str, Any]:
    # Initialize workflow state
    initial_state = {
        "company_name": company_name,
        "tavily_data": {},
        "crm_data": {},
        # ... other fields
    }
    
    # Create and execute LangGraph workflow
    agent = create_agent_graph()
    final_state = agent.invoke(initial_state)
    
    return final_state
```

### **Workflow Creation**
```python
def create_agent_graph():
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("research", research_node)
    workflow.add_node("analyze", analyze_node)
    workflow.add_node("generate_pitch", generate_pitch_node)
    workflow.add_node("score_lead", score_lead_node)
    workflow.add_node("save_results", save_results_node)
    
    # Define execution flow
    workflow.set_entry_point("research")
    workflow.add_edge("research", "analyze")
    workflow.add_edge("analyze", "generate_pitch")
    workflow.add_edge("generate_pitch", "score_lead")
    workflow.add_edge("score_lead", "save_results")
    workflow.add_edge("save_results", END)
    
    return workflow.compile()
```

##  Output Format

### **Final Report Structure**
```
======================================================================
 RESEARCH REPORT: {Company Name}
======================================================================

 ANALYSIS:
- Company overview (size, industry, developments)
- Key decision makers and contacts  
- Recent news and pain points
- Sales opportunities
- Red flags/concerns

──────────────────────────────────────────────────────────────────────

 PERSONALIZED PITCH:
- 3 email subject line options
- Personalized email body (< 150 words)
- Value proposition
- 3-5 key talking points
- Specific action items

──────────────────────────────────────────────────────────────────────

 LEAD SCORE: X.X/10

 ACTION ITEMS:
1. Specific next step 1
2. Specific next step 2
3. Specific next step 3
...
```

##  Key Features

- ** Workflow Orchestration**: LangGraph manages complex multi-step process
- ** AI-Powered Analysis**: Gemini 2.0 Flash for intelligent insights
- ** Multi-Source Research**: Combines web search + CRM data
- ** Personalized Content**: Custom pitches based on research
- ** Lead Qualification**: Automated scoring with reasoning
- **️ Full Observability**: Langfuse tracing for debugging/optimization
- ** Modular Design**: Separated prompts, graceful API failures
- ** Persistent Storage**: Pinecone for historical data

##  Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys

# Run the agent
python agent.py
```

The agent will research "Stripe" by default, but you can modify the `company_to_research` variable to analyze any company.