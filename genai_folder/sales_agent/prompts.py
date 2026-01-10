"""
Sales Agent Prompts Configuration
"""

# System Messages
SALES_ANALYST_SYSTEM = "You are an expert sales intelligence analyst."
SALES_STRATEGIST_SYSTEM = "You are an expert sales strategist specializing in personalized outreach."
SALES_QUALIFIER_SYSTEM = "You are a sales qualification expert."

# Analysis Prompt Template
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

Keep it focused and actionable.
"""

# Pitch Generation Prompt Template
PITCH_PROMPT = """
Based on this analysis of {company_name}:

{analysis}

Create a personalized sales outreach plan including:

1. EMAIL SUBJECT LINES (3 options - compelling and specific)
2. EMAIL BODY (personalized opener referencing their recent news/achievements)
3. VALUE PROPOSITION (how our solution addresses their specific needs)
4. KEY TALKING POINTS (3-5 bullet points for calls)
5. ACTION ITEMS (specific next steps for the sales rep)

Make it highly personalized - reference specific details from the research.
Keep email draft under 150 words.
"""

# Lead Scoring Prompt Template
SCORING_PROMPT = """
Based on this research and analysis:

COMPANY: {company_name}
ANALYSIS: {analysis}

Score this lead from 0-10 and provide reasoning.
Also list 3-5 specific action items for the sales rep.

Format:
SCORE: X/10
REASONING: (2-3 sentences)
ACTION ITEMS:
1. ...
2. ...
3. ...

Consider: company size, recent activity, fit with our solution, contact accessibility, timing.
"""