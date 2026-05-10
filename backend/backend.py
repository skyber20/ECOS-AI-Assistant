from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import json
import os
from datetime import datetime

#from API import agent ili kak tam u nas ya xz
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AgentRequest(BaseModel):
    prompt: str
    context: Optional[dict] = None

class AgentResponse(BaseModel):
    response: str
    context: dict
    tool_results: list

@app.post("/api/agent/run")
async def run_agent(request: AgentRequest):
    # Здесь вызываем этого бобика
    # result = your_agent.run(request.prompt, request.context)
    
    # Заглушка
    result = {
        "response": f"Ответ на: {request.prompt}",
        "context": request.context or {},
        "tool_results": []
    }
    
    log_result(result)
    
    return result

@app.post("/api/agent/continue")
async def continue_agent(request: AgentRequest):
    
    result = {
        "response": f"Продолжение с контекстом",
        "context": request.context or {},
        "tool_results": []
    }
    
    log_result(result)
    return result

@app.get("/api/logs")
async def get_logs():
    logs = []
    if os.path.exists("agent_logs"):
        for file in os.listdir("agent_logs"):
            with open(f"agent_logs/{file}", "r") as f:
                logs.append(json.load(f))
    return logs

def log_result(result: dict):
    os.makedirs("agent_logs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    with open(f"agent_logs/{timestamp}.json", "w") as f:
        json.dump(result, f, indent=2, default=str)
