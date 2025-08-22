from fastapi import FastAPI, HTTPException, Depends, Header
# from typing import Optional
# from config import API_KEY
from models import RunRequest, RunResponse
from services import run_rag
from fastapi.middleware.cors import CORSMiddleware
import os, uvicorn
import asyncio
import aiohttp
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

origins = [
    "*"
]

app = FastAPI(title="RAG", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variable to track ping status
ping_status = {"last_ping": None, "ping_count": 0, "is_running": False}

async def ping_google():
    """Ping Google every 2 minutes to keep the app alive on Render"""
    global ping_status
    
    ping_status["is_running"] = True
    ping_status["ping_count"] = 0
    
    while ping_status["is_running"]:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get("https://www.google.com", timeout=10) as response:
                    if response.status == 200:
                        ping_status["last_ping"] = datetime.now().isoformat()
                        ping_status["ping_count"] += 1
                        logger.info(f"Ping successful! Count: {ping_status['ping_count']}, Time: {ping_status['last_ping']}")
                    else:
                        logger.warning(f"Ping failed with status: {response.status}")
        except Exception as e:
            logger.error(f"Ping error: {e}")
        
        # Wait for 2 minutes (120 seconds)
        await asyncio.sleep(120)

@app.on_event("startup")
async def startup_event():
    """Start the ping task when the app starts"""
    asyncio.create_task(ping_google())
    logger.info("Ping task started")

@app.on_event("shutdown")
async def shutdown_event():
    """Stop the ping task when the app shuts down"""
    ping_status["is_running"] = False
    logger.info("Ping task stopped")

# # Authentication dependency
# def verify_api_key(authorization: Optional[str] = Header(None)):
#     if not authorization or not authorization.startswith("Bearer "):
#         raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
    
#     token = authorization.split(" ")[1]
#     if token != API_KEY:
#         if token != API_KEY:
#             raise HTTPException(status_code=403, detail="Invalid API key")
#     return True

@app.post("/api/v1/hackrx/run", response_model=RunResponse)
async def hackrx_run(payload: RunRequest):
    try:
        answers = await run_rag(str(payload.documents), payload.questions)
        return RunResponse(answers=answers)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {"message": "Hello World"}

@app.get("/greet/{name}")
def greet(name: str):
    return {"message": f"Hello, {name}!"}

@app.get("/health")
def health_check():
    """Health check endpoint to monitor the app status"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "ping_status": ping_status
    }

@app.get("/ping")
def ping_info():
    """Get current ping status"""
    return ping_status

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
