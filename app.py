import logging
import os
import re
import smtplib
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import bcrypt
import jwt
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from pydantic import BaseModel, EmailStr
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer, util

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env
load_dotenv()

# Set API keys for LangChain/Groqx
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
app_password = os.getenv("app_password")


SECRET_KEY = os.getenv("SECRET_KEY", "your_secret_key")
ALGORITHM = "HS256"


# Initialize FastAPI
app = FastAPI()

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MongoDB Setup
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
client = MongoClient(MONGO_URI)
db = client.faculty_recruitment
users = db.users
collection = db.resumes

# ---------------------------
# Models for Authentication
# ---------------------------
class SignupRequest(BaseModel):
    name: str
    email: EmailStr
    password: str

class LoginRequest(BaseModel):
    email: EmailStr
    password: str

from fastapi import Depends, Header


def get_current_user(authorization: str = Header(...)):
    """
    Extracts the logged-in user's email from the JWT token.
    """
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    try:
        token = authorization.split(" ")[1]  # Extract token from "Bearer <token>"
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload.get("email")
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid token")

def get_current_admin(authorization: str = Header(...)):
    """
    Dependency that verifies a valid JWT token is provided
    and that the user has the "admin" role.
    """
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    try:
        # Expect header format "Bearer <token>"
        token = authorization.split(" ")[1]
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        if payload.get("role") != "admin":
            raise HTTPException(status_code=403, detail="Not authorized as admin")
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid token")


# ---------------------------
# Signup Endpoint
# ---------------------------
@app.post("/signup")
async def signup(request: SignupRequest):
    if users.find_one({"email": request.email}):
        raise HTTPException(status_code=400, detail="Email already registered")
    
    # Hash the password before storing
    hashed_pw = bcrypt.hashpw(request.password.encode('utf-8'), bcrypt.gensalt())
    
    # Insert the new user document into MongoDB
    users.insert_one({
        "name": request.name,
        "email": request.email,
        "password": hashed_pw,
        "role": "user"
    })
    return {"success": True, "message": "User registered successfully"}

# ---------------------------
# Login Endpoint
# ---------------------------
@app.post("/login")
async def login(request: LoginRequest):
    user = users.find_one({"email": request.email})
    if not user:
        raise HTTPException(status_code=400, detail="User not found")
    
    if not bcrypt.checkpw(request.password.encode('utf-8'), user["password"]):
        raise HTTPException(status_code=400, detail="Incorrect password")
    
    # Generate a JWT token (valid for 1 day)
    role = user.get("role", "user")
    token_data = {"email": request.email, "role": role, "exp": datetime.utcnow() + timedelta(days=1)}
    token = jwt.encode(token_data, SECRET_KEY, algorithm=ALGORITHM)
    return {"success": True, "message": "Login successful", "token": token}


# Ensure upload directory exists
UPLOAD_DIR = "Uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Initialize LLM and output parser
llm = ChatGroq(model="mixtral-8x7b-32768")
output_parser = StrOutputParser()

# Define prompt template for resume extraction
template1 = """
Extract the following details from the given faculty resume and return the output in a structured text format:

Skills:
- <List of skills>

Experience:
- <Role> at <Institution>, Department: <Department>, Duration: <Years>
- <Role> at <Institution>, Department: <Department>, Duration: <Years>

Education:
- Degree: <Degree>, University: <University>, Graduation Year: <Year>, CGPA: <CGPA>

Publications:
- Title: <Title>, Published in: <Journal>, Year: <Year>
- Title: <Title>, Published in: <Journal>, Year: <Year>

Resume:
{resume_text}
"""

template2='''Extract only the email address from the given resume text and return it as plain text without any extra words or formatting.

Resume:
{resume_text}
'''
prompt2=PromptTemplate(input_variables=['resume_text'],
                       template=template2)

chains=prompt2|llm|output_parser


# Create prompt template chain
prompt = PromptTemplate(input=['resume_text'], template=template1)
chain = prompt | llm | output_parser

# Initialize Sentence Transformer for similarity matching
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Predefined job description for similarity check
JOB_DESCRIPTION='''
We are looking for a passionate and skilled faculty member in Computer Science, AI, and Data Science with expertise in Machine Learning, Python, AI tools, and Fullstack Development. The ideal candidate should have hands-on experience in C++, SQL, Web Technologies (HTML/CSS/JavaScript), and Python libraries for data science applications,
B.Tech in Artificial Intelligence & Data Science / CSE / IT (or related field),CGPA above 8.0 (preferred),Prior experience in AI, ML, and Data Science (internships or projects),
Strong programming skills in Python, C++, SQL, and Fullstack Web Development
Hands-on experience with Python libraries (NumPy, Pandas, TensorFlow, etc.),Knowledge of Database Management (SQL/MySQL),Familiarity with AI tools, deep learning frameworks, and cloud-based AI platforms
Experience in Fullstack Development (HTML, CSS, JavaScript),Teach AI, Machine Learning, and Data Science courses to undergraduate students,Guide students in projects, research, and industry collaborations,Develop AI-powered applications using Python, SQL, and Web Technologies,Research and optimize ML models for real-world applications
'''

def extract_text_from_pdf(file_path: str) -> str:
    """
    Extracts text from a PDF file using PyPDFLoader.
    """
    try:
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        text = "\n".join([doc.page_content for doc in docs])
        return text
    except Exception as e:
        logger.error("Error extracting text from PDF: %s", e)
        raise HTTPException(status_code=500, detail="Error processing PDF file.")

def clean_text(texts: str) -> str:
    """
    Cleans and normalizes text by removing newlines and extra spaces.
    """
    texts = re.sub(r'\n', ' ', texts)
    return texts.strip()

# ---------------------------
# Resume Upload Endpoint
# ---------------------------
@app.post("/upload")
async def upload_resume(file: UploadFile = File(...)):
    """
    Endpoint to upload and process a resume PDF.
    - Saves the file to disk.
    - Extracts text from the PDF.
    - Processes the text using LangChain and computes a match score.
    - Stores the processed resume data in MongoDB.
    """
    try:
        # Save the uploaded file
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        logger.info("Saved file %s", file.filename)

        # Extract text from the PDF file
        raw_text = extract_text_from_pdf(file_path)
        logger.info("Extracted raw text from %s", file.filename)

        # Process the resume text using LLM chain
        response_text = chain.invoke({'resume_text': raw_text})
        user_email=chains.invoke({'resume_text': raw_text})
        response_text = clean_text(response_text)
        logger.info("Processed resume text.")

        # Compute similarity with the predefined job description
        encoded_resume = embedding_model.encode(response_text)
        encoded_job = embedding_model.encode(JOB_DESCRIPTION)
        similarity = util.pytorch_cos_sim(encoded_resume, encoded_job).item()
        logger.info("Computed similarity score: %f", similarity)

        # Store processed resume data in MongoDB
        doc = {
            "filename": file.filename, 
            "resume_text": response_text, 
            "match_score": similarity,
            "uploaded_at": datetime.utcnow()
            }
        collection.insert_one(doc)
        logger.info("Stored resume data in MongoDB.")

        if similarity > 0.6:
            await send_email(user_email)

        return JSONResponse(content={"resume_text": response_text, "match_score": similarity})
    except Exception as e:
        logger.exception("Error in /upload endpoint.")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/send-email")
async def send_email(candidate_email: str):
    """
    Endpoint to send an interview invitation email to a candidate.
    """
    try:
        SMTP_SERVER = "smtp.gmail.com"
        SMTP_PORT = 587
        EMAIL_SENDER = "your_email@email.com"
        EMAIL_PASSWORD = app_password

        msg = MIMEMultipart()
        msg["From"] = EMAIL_SENDER
        msg["To"] = candidate_email
        msg["Subject"] = "Interview Invitation for Faculty at Chennai Institute of Technology"

        Interview_Date = "TBD"  # Update with an actual interview date as needed
        body = f"""
Dear Candidate,

We are pleased to invite you for an interview for the faculty position at Chennai Institute of Technology on {Interview_Date}. Please confirm your availability by replying to this email.

Best regards,
Chennai Institute of Technology
"""
        msg.attach(MIMEText(body, "plain"))

        # Send the email
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(EMAIL_SENDER, EMAIL_PASSWORD)
        server.sendmail(EMAIL_SENDER, candidate_email, msg.as_string())
        server.quit()
        logger.info("Email sent to %s", candidate_email)

        return JSONResponse(content={"message": "Email sent successfully!"})
    except Exception as e:
        logger.exception("Error sending email.")
        raise HTTPException(status_code=500, detail=str(e))
