from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import json
import os
import sys
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'agent'))

from orchestrator import (
    OrchestrationStatus,
    OrchestrationResult,
    ClarificationAnswer,
    LangGraphResearchAgent,
    prepare_intent_for_design,
)
from intent_parser import ResearchIntent

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Глобальное хранилище для отслеживания сессий
sessions: dict = {}


class AgentRequest(BaseModel):
    prompt: str
    session_id: Optional[str] = None  # Добавляем ID сессии
    context: Optional[dict] = None


class ClarificationRequest(BaseModel):
    session_id: Optional[str] = None  # Добавляем ID сессии
    answers: Optional[List[dict]] = None


class AgentResponse(BaseModel):
    status: str
    response: Optional[str] = None
    result: Optional[dict] = None
    clarification_requests: Optional[List[dict]] = None
    error: Optional[str] = None
    session_id: Optional[str] = None  # Возвращаем ID сессии


def get_or_create_session(session_id: Optional[str] = None) -> tuple[str, dict]:
    """Получить или создать сессию"""
    if session_id and session_id in sessions:
        return session_id, sessions[session_id]
    
    # Создаем новую сессию
    import uuid
    new_id = session_id or str(uuid.uuid4())
    sessions[new_id] = {
        "agent": LangGraphResearchAgent(),
        "intent": None,
        "ready": False,
        "current_result": None
    }
    return new_id, sessions[new_id]


def process_orchestration_result(result: OrchestrationResult, session_id: str) -> AgentResponse:
    """Обработка результата от оркестратора"""
    session = sessions[session_id]
    session["current_result"] = result
    
    logger.info(f"Processing result with status: {result.status}")
    logger.info(f"Intent: {result.intent}")
    logger.info(f"Design: {result.research_design}")
    
    # Сохраняем intent для будущих запросов
    if result.intent:
        session["intent"] = result.intent
    
    if result.status == OrchestrationStatus.NEEDS_CLARIFICATION:
        return AgentResponse(
            status="needs_clarification",
            session_id=session_id,
            clarification_requests=[
                req.model_dump(mode="json") for req in result.clarification_requests
            ],
            response=result.message or "Необходимо уточнение",
        )
    
    if result.status == OrchestrationStatus.DESIGN_READY or result.status == OrchestrationStatus.BUILD_PLAN_READY:
        response_data = {
            "session_id": session_id,
            "status": result.status.value,
            "response": result.message or "Дизайн исследования готов.",
        }
        
        if result.research_design:
            response_data["result"] = result.research_design.model_dump(
                mode="json", exclude={"original_query", "blocking_reasons"}
            )
        
        if result.clarification_requests:
            response_data["clarification_requests"] = [
                req.model_dump(mode="json") for req in result.clarification_requests
            ]
            
        return AgentResponse(**response_data)
    
    return AgentResponse(
        session_id=session_id,
        status=result.status.value,
        response=result.message or str(result.status.value),
    )


@app.post("/api/agent/run")
async def run_agent(request: AgentRequest):
    """Запуск нового исследования"""
    try:
        session_id, session = get_or_create_session(request.session_id)
        
        logger.info(f"Starting new research for session {session_id}")
        logger.info(f"Prompt: {request.prompt}")
        
        # Получаем агента для этой сессии
        agent = session["agent"]
        
        # Запускаем flow
        result = agent.run(
            query=request.prompt,
            use_defaults=request.context.get("use_defaults", False) if request.context else False
        )
        
        logger.info(f"Got result: {result}")
        
        if result is None:
            return AgentResponse(
                status="error",
                error="Оркестратор вернул пустой результат",
                session_id=session_id
            )
        
        return process_orchestration_result(result, session_id)
        
    except Exception as exc:
        logger.error(f"Error in run_agent: {exc}", exc_info=True)
        return AgentResponse(
            status="error",
            error=str(exc),
            session_id=request.session_id
        )


@app.post("/api/agent/clarify")
async def clarify_agent(request: ClarificationRequest):
    """Обработка ответов на уточнения"""
    try:
        if not request.session_id:
            raise HTTPException(status_code=400, detail="session_id обязателен")
            
        if request.session_id not in sessions:
            raise HTTPException(status_code=400, detail="Сессия не найдена")
        
        session = sessions[request.session_id]
        agent = session["agent"]
        intent = session.get("intent")
        
        if not intent:
            raise HTTPException(status_code=400, detail="Нет активного исследования")
        
        logger.info(f"Clarifying for session {request.session_id}")
        
        if request.answers:
            # Конвертируем ответы
            answers = [
                ClarificationAnswer(**answer) for answer in request.answers
            ]
            
            # Обновляем intent с ответами
            refined_intent = agent.refine_intent(intent, answers)
            session["intent"] = refined_intent
            
            # Продолжаем flow с уточненным intent
            result = agent.continue_from_intent(refined_intent)
        else:
            # Продолжаем без ответов
            result = agent.continue_from_intent(intent)
        
        logger.info(f"Clarification result: {result}")
        
        return process_orchestration_result(result, request.session_id)
        
    except Exception as exc:
        logger.error(f"Error in clarify_agent: {exc}", exc_info=True)
        return AgentResponse(
            status="error",
            error=str(exc),
            session_id=request.session_id
        )


@app.post("/api/agent/continue")
async def continue_agent(request: AgentRequest):
    """Продолжение исследования"""
    try:
        session_id = request.session_id
        if not session_id or session_id not in sessions:
            return AgentResponse(
                status="error",
                error="Сессия не найдена. Используйте /api/agent/run для начала.",
                session_id=session_id
            )
        
        session = sessions[session_id]
        agent = session["agent"]
        intent = session.get("intent")
        
        if not intent:
            return AgentResponse(
                status="error",
                error="Нет активного исследования",
                session_id=session_id
            )
        
        logger.info(f"Continuing research for session {session_id}")
        
        result = agent.continue_from_intent(
            intent,
            use_defaults=request.context.get("use_defaults", False) if request.context else False
        )
        
        logger.info(f"Continue result: {result}")
        
        return process_orchestration_result(result, session_id)
        
    except Exception as exc:
        logger.error(f"Error in continue_agent: {exc}", exc_info=True)
        return AgentResponse(
            status="error",
            error=str(exc),
            session_id=request.session_id
        )


@app.get("/api/agent/status/{session_id}")
async def get_status(session_id: str):
    """Получить статус сессии"""
    if session_id not in sessions:
        return {"status": "not_found"}
    
    session = sessions[session_id]
    return {
        "session_id": session_id,
        "has_intent": session["intent"] is not None,
        "result_status": session["current_result"].status.value if session["current_result"] else "none"
    }


@app.get("/api/logs")
async def get_logs():
    """Получить логи"""
    logs = []
    if os.path.exists("agent_logs"):
        for file in os.listdir("agent_logs"):
            with open(f"agent_logs/{file}", "r") as f:
                logs.append(json.load(f))
    return logs