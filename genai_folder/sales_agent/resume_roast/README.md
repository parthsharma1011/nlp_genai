# Resume Roaster & Improver 

An AI-powered resume critique and improvement system using CrewAI that brutally analyzes your resume, suggests improvements, and tailors it to specific job postings.

##  What It Does

1. ** Roasts Your Resume**: Brutally honest critique identifying weaknesses
2. ** Improves Everything**: Rewrites sections with better language and structure  
3. ** Job Matching**: Tailors your resume to specific job postings
4. ** ATS Optimization**: Ensures your resume passes applicant tracking systems

##  Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Resume Critic  │───▶│ Resume Improver │───▶│  Job Matcher    │
│                 │    │                 │    │                 │
│ • Brutal honest │    │ • Rewrite weak  │    │ • Tailor to job │
│   feedback      │    │   sections      │    │ • Add keywords  │
│ • Identify gaps │    │ • Add metrics   │    │ • Optimize ATS  │
│ • Rate 1-10     │    │ • Fix format    │    │ • Highlight fit │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

##  Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set Up Environment
Your `.env` file is already configured with:
- Google Gemini API (for AI analysis)
- Tavily API (for job market research)

### 3. Run the Roaster
```bash
python resume_roaster.py
```

##  File Structure

```
resume_roast/
├── resume_roaster.py      # Main CrewAI application
├── requirements.txt       # Python dependencies
├── .env                  # API keys (already configured)
├── README.md             # This file
└── explanations.txt      # Detailed explanations
```

##  The AI Crew

### **Resume Critic** 
- **Role**: Brutal but constructive feedback
- **Goal**: Identify every weakness and improvement opportunity
- **Output**: Detailed critique with 1-10 rating

### **Resume Improver** 
- **Role**: Transform weak resumes into compelling documents
- **Goal**: Rewrite sections with impact-focused language
- **Output**: Completely improved resume

### **Job Matcher** 
- **Role**: Tailor resumes to specific job postings
- **Goal**: Maximize relevance and keyword matching
- **Output**: Job-specific tailored resume

##  Sample Output

The system will generate:

1. **Critique Report**: Detailed analysis of weaknesses
2. **Improved Resume**: Professional rewrite with better formatting
3. **Tailored Version**: Job-specific customization
4. **Change Log**: Explanation of all modifications made

##  Customization

### Add Your Own Resume
Replace the `SAMPLE_RESUME` variable with your actual resume text.

### Target Specific Jobs
Update the `SAMPLE_JOB_POSTING` variable with the job you're applying for.

### Upload Files (Future Enhancement)
The code includes imports for PDF and DOCX parsing:
```python
import PyPDF2
import docx2txt
```

##  Key Features

- **Multi-Agent Collaboration**: Three specialized AI agents work together
- **Sequential Processing**: Each agent builds on the previous one's work
- **Gemini 2.0 Flash**: Latest Google AI for intelligent analysis
- **ATS Optimization**: Ensures compatibility with applicant tracking systems
- **Quantified Improvements**: Adds metrics and measurable achievements
- **Job-Specific Tailoring**: Customizes resume for each application

##  Expected Improvements

- **Structure**: Professional formatting and organization
- **Language**: Strong action verbs and impact statements
- **Metrics**: Quantified achievements and results
- **Keywords**: Industry-relevant terms for ATS systems
- **Relevance**: Tailored content for specific job requirements

##  Usage Examples

### Basic Usage
```python
# Run with sample data
python resume_roaster.py
```

### With Custom Resume
```python
# Modify SAMPLE_RESUME variable in the code
SAMPLE_RESUME = "Your actual resume text here..."
```

### With Specific Job
```python
# Update SAMPLE_JOB_POSTING variable
SAMPLE_JOB_POSTING = "Actual job posting you're applying for..."
```

##  Output Files

- `resume_analysis_results.txt`: Complete analysis and improved resume
- Console output: Real-time progress and agent conversations

##  Future Enhancements

- File upload interface (PDF/DOCX)
- Web interface for easy use
- Multiple resume versions for different job types
- Industry-specific optimization
- Cover letter generation
- Interview preparation based on resume