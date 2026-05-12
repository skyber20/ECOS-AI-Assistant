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

sessions: dict = {}


class AgentRequest(BaseModel):
    prompt: str
    session_id: Optional[str] = None 
    context: Optional[dict] = None


class ClarificationRequest(BaseModel):
    session_id: Optional[str] = None 
    answers: Optional[List[dict]] = None


class AgentResponse(BaseModel):
    status: str
    response: Optional[str] = None
    result: Optional[dict] = None
    clarification_requests: Optional[List[dict]] = None
    error: Optional[str] = None
    session_id: Optional[str] = None 


def get_or_create_session(session_id: Optional[str] = None) -> tuple[str, dict]:
    """Получить или создать сессию"""
    if session_id and session_id in sessions:
        return session_id, sessions[session_id]
    
    import uuid
    new_id = session_id or str(uuid.uuid4())
    sessions[new_id] = {
        "agent": LangGraphResearchAgent(),
        "intent": None,
        "ready": False,
        "current_result": None
    }
    return new_id, sessions[new_id]


def extract_full_result(result: OrchestrationResult) -> dict:
    """
    Извлекает полный результат в формате для фронтенда.
    Возвращает словарь со всеми компонентами исследования.
    """
    data = {}
    
    # Всегда включаем базовую информацию
    if result.intent:
        data["intent"] = {
            "original_query": result.intent.original_query,
            "intent_type": result.intent.intent_type.value if result.intent.intent_type else None,
            "complexity": result.intent.complexity.value if result.intent.complexity else None,
            "topic": result.intent.topic,
            "geography": result.intent.geography,
            "indicators": result.intent.indicators,
            "frequency": result.intent.frequency,
        }
        if result.intent.time_range:
            data["intent"]["time_range"] = result.intent.time_range.model_dump(mode="json")
    
    # Дизайн исследования
    if result.research_design:
        data["research_design"] = result.research_design.model_dump(
            mode="json", 
            exclude={"original_query", "blocking_reasons"}
        )
    
    # Структура датасета
    if result.dataset_structure:
        data["dataset_structure"] = result.dataset_structure.model_dump(mode="json")
    
    # Скрипт сборки
    if result.build_script:
        data["build_script"] = result.build_script.model_dump(mode="json")
    
    # Результаты запуска
    if result.build_run:
        build_run_data = result.build_run.model_dump(mode="json")
        # Извлекаем только нужные поля для фронтенда
        data["build_run"] = {
            "status": build_run_data.get("status"),
            "attempts_count": len(build_run_data.get("attempts", [])),
            "output": build_run_data.get("output"),
            "final_error": build_run_data.get("final_error"),
            "output_dir": build_run_data.get("output_dir"),
        }
    
    # Результаты ранжирования датасетов
    if result.dataset_rerank:
        data["dataset_rerank"] = {
            "results": [
                {
                    "record_id": r.record_id,
                    "dataset_id": r.dataset_id,
                    "title": r.title,
                    "source": r.source,
                    "description": r.description,
                    "relevance": r.relevance.value,
                    "usefulness_confidence": r.usefulness_confidence.value,
                    "why_matched": r.why_matched,
                    "possible_limitations": r.possible_limitations,
                    "data_path": r.data_path,
                    "source_url": r.source_url,
                    "tags": r.tags,
                    "unit": r.unit,
                    "frequency": r.frequency,
                }
                for r in result.dataset_rerank.results
            ],
            "no_results_reason": result.dataset_rerank.no_results_reason,
        }
    
    # Похожие, но отклоненные кандидаты
    if result.dataset_rerank and result.dataset_rerank.rejected_similar_candidates:
        data["rejected_candidates"] = [
            {
                "record_id": r.record_id,
                "dataset_id": r.dataset_id,
                "title": r.title,
                "reason": r.reason,
            }
            for r in result.dataset_rerank.rejected_similar_candidates
        ]
    
    # Датасеты для Explorer
    if result.explorer_datasets:
        data["explorer_datasets"] = [
            item.model_dump(mode="json") for item in result.explorer_datasets
        ]
    
    # Уточняющие вопросы
    if result.clarification_requests:
        data["clarification_requests"] = [
            {
                "field": req.field,
                "reason": req.reason,
                "question": req.question,
                "default_assumption": req.default_assumption,
            }
            for req in result.clarification_requests
        ]
    
    # Сообщение
    if result.message:
        data["message"] = result.message
    
    return data


def process_orchestration_result(result: OrchestrationResult, session_id: str) -> AgentResponse:
    """Обработка результата от оркестратора"""
    session = sessions[session_id]
    session["current_result"] = result
    
    if result.intent:
        session["intent"] = result.intent
    
    # Обработка разных статусов
    if result.status == OrchestrationStatus.NEEDS_CLARIFICATION:
        return AgentResponse(
            status="needs_clarification",
            session_id=session_id,
            clarification_requests=[
                {
                    "field": req.field,
                    "reason": req.reason,
                    "question": req.question,
                    "default_assumption": req.default_assumption,
                }
                for req in result.clarification_requests
            ],
            response=result.message or "Необходимы уточнения для продолжения исследования",
            result=extract_full_result(result)  # Всегда включаем результат
        )
    
    elif result.status == OrchestrationStatus.DESIGN_READY:
        # КЛЮЧЕВОЕ ИЗМЕНЕНИЕ: передаём полный результат
        return AgentResponse(
            status="design_ready",
            session_id=session_id,
            response=result.message or "Исследование завершено. Дизайн и скрипт готовы.",
            result=extract_full_result(result)  # Все компоненты исследования
        )
    
    elif result.status == OrchestrationStatus.BUILD_FAILED:
        return AgentResponse(
            status="build_failed",
            session_id=session_id,
            response=result.message or "Скрипт сборки не удалось выполнить успешно.",
            result=extract_full_result(result)
        )
    
    elif result.status == OrchestrationStatus.NO_DATA:
        return AgentResponse(
            status="no_data",
            session_id=session_id,
            response=result.message or "Данные не найдены в доступных источниках.",
            result=extract_full_result(result)
        )
    
    elif result.status == OrchestrationStatus.UNSUPPORTED:
        return AgentResponse(
            status="unsupported",
            session_id=session_id,
            response=result.message or "Данный тип запроса не поддерживается.",
            result=extract_full_result(result)
        )
    
    elif result.status == OrchestrationStatus.READY_FOR_DESIGN:
        return AgentResponse(
            status="ready_for_design",
            session_id=session_id,
            response=result.message or "Запрос обработан, готов к дизайну исследования.",
            result=extract_full_result(result)
        )
    
    # По умолчанию
    return AgentResponse(
        status=result.status.value,
        session_id=session_id,
        response=result.message or f"Статус: {result.status.value}",
        result=extract_full_result(result)
    )


@app.post("/api/agent/run")
async def run_agent(request: AgentRequest):
    """Запуск нового исследования"""
    try:
        session_id, session = get_or_create_session(request.session_id)
        
        logger.info(f"Starting new research for session {session_id}")
        logger.info(f"Prompt: {request.prompt}")
        
        agent = session["agent"]
        
        # Сбрасываем предыдущие результаты
        session["current_result"] = None
        session["intent"] = None
        
        result = agent.run(
            query=request.prompt,
            use_defaults=request.context.get("use_defaults", False) if request.context else False
        )
        
        if result is None:
            return AgentResponse(
                status="error",
                error="Оркестратор вернул пустой результат. Попробуйте переформулировать запрос.",
                session_id=session_id
            )
        
        logger.info(f"Result status: {result.status}")
        if result.intent:
            logger.info(f"Intent type: {result.intent.intent_type}")
        if result.research_design:
            logger.info("Research design available")
        if result.build_script:
            logger.info("Build script available")
        if result.build_run:
            logger.info(f"Build run status: {result.build_run.status}")
        
        return process_orchestration_result(result, session_id)
        
    except Exception as exc:
        logger.error(f"Error in run_agent: {exc}", exc_info=True)
        return AgentResponse(
            status="error",
            error=f"Внутренняя ошибка: {str(exc)}",
            session_id=request.session_id
        )


@app.post("/api/agent/clarify")
async def clarify_agent(request: ClarificationRequest):
    """Обработка ответов на уточнения"""
    try:
        if not request.session_id:
            raise HTTPException(status_code=400, detail="session_id обязателен")
            
        if request.session_id not in sessions:
            raise HTTPException(status_code=400, detail="Сессия не найдена. Начните с /api/agent/run")
        
        session = sessions[request.session_id]
        agent = session["agent"]
        intent = session.get("intent")
        
        if not intent:
            raise HTTPException(status_code=400, detail="Нет активного исследования для уточнения")
        
        if request.answers:
            answers = [
                ClarificationAnswer(**answer) for answer in request.answers
            ]
            refined_intent = agent.refine_intent(intent, answers)
            session["intent"] = refined_intent
            result = agent.continue_from_intent(refined_intent)
        else:
            result = agent.continue_from_intent(intent)
        
        return process_orchestration_result(result, request.session_id)
        
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Error in clarify_agent: {exc}", exc_info=True)
        return AgentResponse(
            status="error",
            error=str(exc),
            session_id=request.session_id
        )


@app.post("/api/agent/continue")
async def continue_agent(request: AgentRequest):
    """Продолжение исследования с текущим intent"""
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
                error="Нет активного исследования для продолжения",
                session_id=session_id
            )
        
        result = agent.continue_from_intent(
            intent,
            use_defaults=request.context.get("use_defaults", False) if request.context else False
        )
        
        return process_orchestration_result(result, session_id)
        
    except Exception as exc:
        logger.error(f"Error in continue_agent: {exc}", exc_info=True)
        return AgentResponse(
            status="error",
            error=str(exc),
            session_id=request.session_id
        )


@app.get("/api/agent/result/{session_id}")
async def get_result(session_id: str):
    """Получить детальный результат сессии"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Сессия не найдена")
    
    session = sessions[session_id]
    result = session.get("current_result")
    
    if not result:
        return {"session_id": session_id, "status": "no_result"}
    
    return {
        "session_id": session_id,
        "status": result.status.value,
        "result": extract_full_result(result)
    }


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