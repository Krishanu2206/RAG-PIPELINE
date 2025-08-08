
# main.py
from fastapi import FastAPI, HTTPException, Depends, Header
# from typing import Optional
# from config import API_KEY
from models import RunRequest, RunResponse
from services import run_rag
from fastapi.middleware.cors import CORSMiddleware

origins = [
    "*"
]

app = FastAPI(title="HackRx API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# # Authentication dependency
# def verify_api_key(authorization: Optional[str] = Header(None)):
#     if not authorization or not authorization.startswith("Bearer "):
#         raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
    
#     token = authorization.split(" ")[1]
#     if token != API_KEY:
#         raise HTTPException(status_code=403, detail="Invalid API key")
#     return True

@app.post("/api/v1/hackrx/run", response_model=RunResponse)
def hackrx_run(payload: RunRequest):
    try:
        answers = run_rag(payload.documents, payload.questions)
        return RunResponse(answers=answers)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/")
def read_root():
    return {"message": "Hello World"}

@app.get("/greet/{name}")
def greet(name: str):
    return {"message": f"Hello, {name}!"}