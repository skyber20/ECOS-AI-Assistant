from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import json
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'agent'))

from orchestrator import (
    OrchestrationStatus,
    OrchestrationResult,
    ClarificationAnswer,
    run_research_flow,
    continue_research_flow,
    refine_intent_with_clarifications,
    prepare_intent_for_design,
)
from intent_parser import ResearchIntent

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Храним состояние исследования
research_state: dict = {}


class AgentRequest(BaseModel):
    prompt: str
    context: Optional[dict] = None


class ClarificationRequest(BaseModel):
    answers: Optional[List[dict]] = None


class AgentResponse(BaseModel):
    status: str
    response: Optional[str] = None
    result: Optional[dict] = None
    clarification_requests: Optional[List[dict]] = None
    error: Optional[str] = None


@app.post("/api/agent/run")
async def run_agent(request: AgentRequest):
    global research_state
    try:
        result: OrchestrationResult = run_research_flow(
            query=request.prompt,
        )
        research_state = {
            "intent": result.intent,
            "design": result.research_design,
            "status": result.status.value,
            "clarification_requests": [
                req.model_dump(mode="json") for req in result.clarification_requests
            ],
        }

        if result.status == OrchestrationStatus.NEEDS_CLARIFICATION:
            return AgentResponse(
                status="needs_clarification",
                clarification_requests=[
                    req.model_dump(mode="json") for req in result.clarification_requests
                ],
                response=result.message,
            )

        if result.status == OrchestrationStatus.DESIGN_READY:
            return AgentResponse(
                status="design_ready",
                result=result.research_design.model_dump(
                    mode="json", exclude={"original_query", "blocking_reasons"}
                ),
                response="Дизайн исследования готов.",
            )

        return AgentResponse(
            status=result.status.value,
            response=result.message or result.status.value,
        )

    except Exception as exc:
        return AgentResponse(status="error", error=str(exc))


@app.post("/api/agent/clarify")
async def clarify_agent(request: ClarificationRequest):
    global research_state
    try:
        intent = research_state.get("intent")
        if not intent:
            raise HTTPException(status_code=400, detail="Нет активного исследования.")

        if request.answers:
            answers = [
                ClarificationAnswer(**answer) for answer in request.answers
            ]
            intent = refine_intent_with_clarifications(intent, answers)

        result: OrchestrationResult = continue_research_flow(intent)

        research_state = {
            "intent": result.intent,
            "design": result.research_design,
            "status": result.status.value,
            "clarification_requests": [
                req.model_dump(mode="json") for req in result.clarification_requests
            ],
        }

        if result.status == OrchestrationStatus.NEEDS_CLARIFICATION:
            return AgentResponse(
                status="needs_clarification",
                clarification_requests=[
                    req.model_dump(mode="json") for req in result.clarification_requests
                ],
                response=result.message,
            )

        if result.status == OrchestrationStatus.DESIGN_READY:
            return AgentResponse(
                status="design_ready",
                result=result.research_design.model_dump(
                    mode="json", exclude={"original_query", "blocking_reasons"}
                ),
                response="Дизайн исследования готов.",
            )

        return AgentResponse(
            status=result.status.value,
            response=result.message or result.status.value,
        )

    except Exception as exc:
        return AgentResponse(status="error", error=str(exc))


@app.post("/api/agent/continue")
async def continue_agent(request: AgentRequest):
    global research_state
    try:
        intent = research_state.get("intent")
        if not intent:
            return AgentResponse(status="error", error="Нет активного исследования.")

        result: OrchestrationResult = continue_research_flow(intent)

        research_state = {
            "intent": result.intent,
            "design": result.research_design,
            "status": result.status.value,
            "clarification_requests": [
                req.model_dump(mode="json") for req in result.clarification_requests
            ],
        }

        if result.status == OrchestrationStatus.NEEDS_CLARIFICATION:
            return AgentResponse(
                status="needs_clarification",
                clarification_requests=[
                    req.model_dump(mode="json") for req in result.clarification_requests
                ],
                response=result.message,
            )

        if result.status == OrchestrationStatus.DESIGN_READY:
            return AgentResponse(
                status="design_ready",
                result=result.research_design.model_dump(
                    mode="json", exclude={"original_query", "blocking_reasons"}
                ),
                response="Дизайн исследования готов.",
            )

        return AgentResponse(
            status=result.status.value,
            response=result.message or result.status.value,
        )

    except Exception as exc:
        return AgentResponse(status="error", error=str(exc))


@app.get("/api/logs")
async def get_logs():
    logs = []
    if os.path.exists("agent_logs"):
        for file in os.listdir("agent_logs"):
            with open(f"agent_logs/{file}", "r") as f:
                logs.append(json.load(f))
    return logs