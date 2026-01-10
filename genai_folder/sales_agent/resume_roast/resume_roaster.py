import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process, LLM
import PyPDF2
import docx2txt

# Load environment variables
load_dotenv()

# Initialize LLM for CrewAI
llm = LLM(
    model="gemini/gemini-2.0-flash",
    api_key=os.getenv("GOOGLE_API_KEY")
)

# Sample resume text (you can replace this with file upload logic)
SAMPLE_RESUME = """
John Doe
Software Engineer
Email: john@email.com | Phone: 123-456-7890

EXPERIENCE:
- Software Developer at ABC Corp (2020-2023)
  - Worked on web applications
  - Used Python and JavaScript
  - Fixed bugs and added features

EDUCATION:
- Bachelor's in Computer Science, XYZ University (2016-2020)

SKILLS:
- Python, JavaScript, HTML, CSS
- Some experience with databases
"""

SAMPLE_JOB_POSTING = """
Senior Full Stack Developer - Tech Startup
We're looking for a passionate Senior Full Stack Developer to join our growing team.

Requirements:
- 5+ years of experience in full-stack development
- Expertise in React, Node.js, Python, and cloud technologies
- Experience with microservices architecture
- Strong problem-solving skills and leadership experience
- Bachelor's degree in Computer Science or related field

Responsibilities:
- Lead development of scalable web applications
- Mentor junior developers
- Collaborate with product and design teams
- Implement best practices and code reviews

Application Requirements:
- Please submit your resume and a cover letter explaining your interest in this role
- Cover letter must highlight your leadership experience and technical expertise
"""

# Define Agents
resume_critic = Agent(
    role='Resume Critic',
    goal='Brutally honest critique of resumes to identify weaknesses and areas for improvement',
    backstory="""You are a seasoned HR professional and career coach with 15 years of experience. 
    You've seen thousands of resumes and know exactly what makes recruiters reject candidates. 
    You provide harsh but constructive feedback to help people improve their resumes.""",
    verbose=True,
    allow_delegation=False,
    llm=llm
)

resume_improver = Agent(
    role='Resume Improvement Specialist',
    goal='Transform weak resumes into compelling, ATS-friendly documents that get interviews',
    backstory="""You are an expert resume writer who specializes in creating compelling resumes 
    that pass ATS systems and impress hiring managers. You know the latest trends in resume 
    formatting, keyword optimization, and how to quantify achievements effectively.""",
    verbose=True,
    allow_delegation=False,
    llm=llm
)

job_matcher = Agent(
    role='Job Matching Specialist',
    goal='Tailor resumes to specific job postings for maximum relevance and impact',
    backstory="""You are a recruitment expert who understands how to align candidate profiles 
    with job requirements. You excel at identifying key skills and experiences that should be 
    highlighted for specific roles and industries.""",
    verbose=True,
    allow_delegation=False,
    llm=llm
)

cover_letter_writer = Agent(
    role='Cover Letter Writer',
    goal='Create compelling, personalized cover letters that complement the tailored resume',
    backstory="""You are an expert cover letter writer who creates compelling narratives 
    that connect candidate experiences to job requirements. You write engaging cover letters 
    that tell a story and make candidates stand out from the competition.""",
    verbose=True,
    allow_delegation=False,
    llm=llm
)

# Define Tasks
critique_task = Task(
    description=f"""
    Analyze this resume and provide a brutal but constructive critique:
    
    {SAMPLE_RESUME}
    
    Focus on:
    1. Overall structure and formatting issues
    2. Weak or vague language
    3. Missing quantifiable achievements
    4. Skills section problems
    5. Experience descriptions that lack impact
    6. ATS compatibility issues
    7. Grammar and spelling errors
    
    Be specific about what's wrong and why it would hurt the candidate's chances.
    Rate the resume out of 10 and explain the rating.
    """,
    agent=resume_critic,
    expected_output="A detailed critique with specific issues identified and a numerical rating"
)

improvement_task = Task(
    description=f"""
    Based on the critique, rewrite and improve this resume:
    
    {SAMPLE_RESUME}
    
    Improvements should include:
    1. Better formatting and structure
    2. Stronger action verbs and impact-focused language
    3. Quantified achievements where possible
    4. ATS-optimized keywords
    5. Professional summary/objective
    6. Improved skills section
    7. Better experience descriptions
    
    Provide the complete improved resume in a professional format.
    """,
    agent=resume_improver,
    expected_output="A completely rewritten and improved resume with professional formatting"
)

job_matching_task = Task(
    description=f"""
    Tailor the improved resume to this specific job posting:
    
    JOB POSTING:
    {SAMPLE_JOB_POSTING}
    
    Customizations should include:
    1. Highlighting relevant skills and experiences
    2. Adding missing keywords from the job posting
    3. Reordering sections for maximum relevance
    4. Adjusting the professional summary
    5. Emphasizing leadership and mentoring experience
    6. Adding relevant technical skills
    
    IMPORTANT: Also check if this job posting mentions requiring a cover letter.
    If it does, set a flag: COVER_LETTER_REQUIRED: YES
    If not, set: COVER_LETTER_REQUIRED: NO
    
    Provide the final tailored resume and explain the key changes made.
    """,
    agent=job_matcher,
    expected_output="A job-specific tailored resume with explanation of changes made and cover letter requirement flag"
)

cover_letter_task = Task(
    description=f"""
    Create a compelling cover letter based on:
    
    TAILORED RESUME: {{{{ job_matching_task.output }}}}
    JOB POSTING: {SAMPLE_JOB_POSTING}
    
    The cover letter should:
    1. Have an engaging opening that references the specific role
    2. Connect candidate's experience to job requirements
    3. Tell a compelling story about the candidate's career journey
    4. Highlight 2-3 key achievements that align with the role
    5. Show enthusiasm for the company and position
    6. Include a strong closing with call to action
    7. Be professional yet personable in tone
    8. Keep it to 3-4 paragraphs, under 400 words
    
    Format it as a proper business letter.
    """,
    agent=cover_letter_writer,
    expected_output="A professional, compelling cover letter tailored to the job posting"
)

def check_cover_letter_requirement(job_posting):
    """Check if job posting mentions cover letter requirement"""
    cover_letter_keywords = [
        "cover letter", "covering letter", "letter of interest", 
        "motivation letter", "must include cover letter", 
        "cover letter required", "cover letter mandatory",
        "please include a cover letter", "attach cover letter"
    ]
    
    job_posting_lower = job_posting.lower()
    for keyword in cover_letter_keywords:
        if keyword in job_posting_lower:
            return True
    return False

def create_conditional_crew(job_posting):
    """Create crew with conditional cover letter agent"""
    
    # Base agents and tasks
    base_agents = [resume_critic, resume_improver, job_matcher]
    base_tasks = [critique_task, improvement_task, job_matching_task]
    
    # Check if cover letter is required
    needs_cover_letter = check_cover_letter_requirement(job_posting)
    
    if needs_cover_letter:
        print("🔍 Cover letter requirement detected! Adding Cover Letter Writer agent...")
        base_agents.append(cover_letter_writer)
        base_tasks.append(cover_letter_task)
    else:
        print("📝 No cover letter requirement found. Using standard resume workflow...")
    
    return Crew(
        agents=base_agents,
        tasks=base_tasks,
        process=Process.sequential,
        verbose=True
    ), needs_cover_letter

def roast_and_improve_resume():
    """Main function to run the resume roasting and improvement process"""
    print("🔥 RESUME ROASTER & IMPROVER 🔥")
    print("=" * 50)
    
    try:
        # Create conditional crew based on job posting
        resume_crew, has_cover_letter = create_conditional_crew(SAMPLE_JOB_POSTING)
        
        # Execute the crew
        result = resume_crew.kickoff()
        
        # Save results to file
        output_filename = "resume_analysis_results.txt"
        if has_cover_letter:
            output_filename = "resume_and_cover_letter_results.txt"
            
        with open(output_filename, "w") as f:
            f.write("RESUME ROASTER & IMPROVER RESULTS\n")
            f.write("=" * 50 + "\n\n")
            if has_cover_letter:
                f.write("📝 INCLUDES COVER LETTER (Required by job posting)\n\n")
            f.write(str(result))
        
        print(f"\n✅ Analysis complete! Results saved to '{output_filename}'")
        
        if has_cover_letter:
            print("📝 Cover letter was generated as required by the job posting!")
        
        return result
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

if __name__ == "__main__":
    roast_and_improve_resume()