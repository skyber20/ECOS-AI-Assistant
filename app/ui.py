from __future__ import annotations

import json
import mimetypes
import os
from typing import Any

import requests
import streamlit as st


DEFAULT_BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
REQUEST_TIMEOUT = (10, 60 * 60)

STATUS_LABELS = {
    "needs_clarification": "Нужны уточнения",
    "ready_for_design": "Готов к дизайну",
    "design_ready": "Готово",
    "build_failed": "Сборка упала",
    "no_data": "Нет данных",
    "unsupported": "Не поддерживается",
}


def main() -> None:
    st.set_page_config(page_title="ECOS AI Assistant", layout="wide")
    st.title("ECOS AI Assistant")

    settings = _sidebar_settings()
    _render_health(settings["backend_url"])

    with st.form("research_form", clear_on_submit=False):
        query = st.text_area(
            "Запрос",
            value=st.session_state.get("query", ""),
            height=130,
            placeholder="Например: Исследуй ВВП США за 2022 год",
        )
        submitted = st.form_submit_button("Запустить", type="primary")

    if submitted:
        st.session_state["query"] = query
        _start_research(query, settings)

    response = st.session_state.get("response")
    if response:
        _render_response(response, settings)


def _sidebar_settings() -> dict[str, Any]:
    with st.sidebar:
        st.header("Параметры")
        backend_url = st.text_input("Backend URL", value=DEFAULT_BACKEND_URL).rstrip("/")
        provider_choice = st.selectbox("LLM provider", ["из .env", "qwen", "yandex"])
        model = st.text_input("Модель", value="")
        use_defaults = st.checkbox("Default-допущения на старте", value=False)
        run_build_script = st.checkbox("Запускать сборку датасета", value=True)
        max_build_tries = st.number_input("Попытки сборки", min_value=1, max_value=10, value=3)

    return {
        "backend_url": backend_url,
        "provider": None if provider_choice == "из .env" else provider_choice,
        "model": model.strip() or None,
        "use_defaults": use_defaults,
        "run_build_script": run_build_script,
        "max_build_tries": int(max_build_tries),
    }


def _render_health(backend_url: str) -> None:
    try:
        health = _get_json(backend_url, "/health")
    except requests.RequestException as exc:
        st.error(f"Backend недоступен: {exc}")
        return

    missing = []
    if not health.get("catalog_records_exists"):
        missing.append("data/catalog_records.jsonl")
    if not health.get("catalog_bm25_exists"):
        missing.append("data/catalog_bm25.sqlite")
    if missing:
        st.warning("Не найдено: " + ", ".join(missing))


def _start_research(query: str, settings: dict[str, Any]) -> None:
    if not query.strip():
        st.warning("Введите запрос.")
        return

    payload = {
        "query": query,
        "provider": settings["provider"],
        "model": settings["model"],
        "use_defaults": settings["use_defaults"],
        "run_build_script": settings["run_build_script"],
        "max_build_tries": settings["max_build_tries"],
    }
    with st.spinner("Пайплайн работает..."):
        try:
            response = _post_json(settings["backend_url"], "/api/research", payload)
        except requests.RequestException as exc:
            _render_api_error(exc)
            return
    st.session_state["response"] = response
    st.rerun()


def _continue_research(
    response: dict[str, Any],
    settings: dict[str, Any],
    answers: list[dict[str, Any]],
    use_defaults: bool = False,
) -> None:
    payload = {
        "intent": response["result"]["intent"],
        "answers": answers,
        "provider": settings["provider"],
        "model": settings["model"],
        "use_defaults": use_defaults,
        "run_build_script": settings["run_build_script"],
        "max_build_tries": settings["max_build_tries"],
    }
    with st.spinner("Продолжаю пайплайн..."):
        try:
            next_response = _post_json(settings["backend_url"], "/api/research/continue", payload)
        except requests.RequestException as exc:
            _render_api_error(exc)
            return
    st.session_state["response"] = next_response
    st.rerun()


def _render_response(response: dict[str, Any], settings: dict[str, Any]) -> None:
    status = response.get("status", "")
    st.subheader(STATUS_LABELS.get(status, status or "Статус неизвестен"))

    cols = st.columns([1, 2, 3])
    cols[0].metric("Статус", STATUS_LABELS.get(status, status))
    cols[1].text_input("Run ID", value=response.get("run_id", ""), disabled=True)
    cols[2].text_input("Артефакты", value=response.get("artifact_dir", ""), disabled=True)

    message = response.get("message")
    if message:
        st.info(message)

    if status == "needs_clarification":
        _render_clarifications(response, settings)

    report_tab, sources_tab, artifacts_tab, json_tab = st.tabs(
        ["Отчет", "Источники", "Артефакты", "JSON"]
    )
    with report_tab:
        report = response.get("report_markdown")
        if report:
            st.markdown(report)
        else:
            st.info("Отчет пока не сформирован.")
    with sources_tab:
        _render_sources(response)
    with artifacts_tab:
        _render_artifacts(response, settings["backend_url"])
    with json_tab:
        st.json(response.get("result", {}), expanded=False)


def _render_clarifications(response: dict[str, Any], settings: dict[str, Any]) -> None:
    requests_payload = response.get("result", {}).get("clarification_requests") or []
    if not requests_payload:
        return

    st.markdown("### Уточнения")
    with st.form("clarifications_form"):
        answers: list[dict[str, Any]] = []
        missing_fields: list[str] = []
        missing_defaults: list[str] = []

        for index, item in enumerate(requests_payload, start=1):
            default = item.get("default_assumption") or ""
            if not default:
                missing_defaults.append(item.get("field") or str(index))

            st.markdown(f"**{index}. {item.get('question') or f'Уточнение {index}'}**")
            if item.get("reason"):
                st.caption(item["reason"])

            answer = st.text_area(
                "Ответ",
                value=default,
                key=f"clarification_{response.get('run_id')}_{index}",
                label_visibility="collapsed",
            )
            answer_text = answer.strip()
            if not answer_text:
                missing_fields.append(item.get("field") or str(index))
            answers.append(
                {
                    "field": item.get("field") or f"clarification_{index}",
                    "question": item.get("question") or "",
                    "answer": answer_text,
                    "used_default": bool(default and answer_text == default.strip()),
                }
            )

        submit_answers = st.form_submit_button("Продолжить с ответами", type="primary")
        submit_defaults = st.form_submit_button("Принять default-допущения")

    if submit_answers:
        if missing_fields:
            st.warning("Заполните все уточнения.")
            return
        _continue_research(response, settings, answers)

    if submit_defaults:
        if missing_defaults:
            st.warning("Не у всех уточнений есть default-допущения.")
            return
        default_answers = [
            {
                "field": item.get("field") or f"clarification_{index}",
                "question": item.get("question") or "",
                "answer": item.get("default_assumption") or "",
                "used_default": True,
            }
            for index, item in enumerate(requests_payload, start=1)
        ]
        _continue_research(response, settings, default_answers, use_defaults=True)


def _render_sources(response: dict[str, Any]) -> None:
    rerank = response.get("result", {}).get("dataset_rerank") or {}
    results = rerank.get("results") or []
    if not results:
        reason = rerank.get("no_results_reason") or "RAG не вернул выбранные источники."
        st.info(reason)
        return

    for item in results:
        title = item.get("title") or item.get("record_id") or "Источник"
        st.markdown(f"**{title}**")
        details = [
            ("record_id", item.get("record_id")),
            ("dataset_id", item.get("dataset_id")),
            ("source", item.get("source")),
            ("frequency", item.get("frequency")),
            ("unit", item.get("unit")),
        ]
        st.caption(" | ".join(f"{key}: {value}" for key, value in details if value))
        if item.get("why_matched"):
            st.write(item["why_matched"])
        if item.get("possible_limitations"):
            st.warning("; ".join(str(value) for value in item["possible_limitations"]))
        if item.get("data_path"):
            st.code(item["data_path"])
        if item.get("source_url"):
            st.link_button("Открыть source_url", item["source_url"])
        st.divider()

    rejected = rerank.get("rejected_similar_candidates") or []
    if rejected:
        with st.expander("Похожие, но отклоненные источники"):
            for item in rejected:
                title = item.get("title") or item.get("record_id") or "Источник"
                st.markdown(f"- **{title}**: {item.get('reason')}")


def _render_artifacts(response: dict[str, Any], backend_url: str) -> None:
    artifacts = response.get("artifacts") or []
    if not artifacts:
        st.info("Артефактов пока нет.")
        return

    for artifact in artifacts:
        artifact_type = artifact.get("artifact_type", "artifact")
        filename = artifact.get("filename", artifact_type)
        download_url = artifact.get("download_url")
        if not download_url:
            st.write(f"{artifact_type}: {filename}")
            continue
        try:
            data = _download_artifact(backend_url, download_url)
        except requests.RequestException as exc:
            st.error(f"{filename}: {exc}")
            continue
        mime = mimetypes.guess_type(filename)[0] or "application/octet-stream"
        st.download_button(
            label=f"{artifact_type}: {filename}",
            data=data,
            file_name=filename,
            mime=mime,
            key=f"download_{response.get('run_id')}_{artifact_type}",
        )


def _get_json(backend_url: str, path: str) -> dict[str, Any]:
    response = requests.get(f"{backend_url}{path}", timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return response.json()


def _post_json(backend_url: str, path: str, payload: dict[str, Any]) -> dict[str, Any]:
    response = requests.post(f"{backend_url}{path}", json=payload, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return response.json()


@st.cache_data(show_spinner=False)
def _download_artifact(backend_url: str, download_url: str) -> bytes:
    response = requests.get(f"{backend_url}{download_url}", timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return response.content


def _render_api_error(exc: requests.RequestException) -> None:
    detail: Any = str(exc)
    response = getattr(exc, "response", None)
    if response is not None:
        try:
            payload = response.json()
            detail = payload.get("detail", payload)
        except json.JSONDecodeError:
            detail = response.text
    if isinstance(detail, dict):
        st.error(f"{detail.get('type', 'Ошибка')}: {detail.get('message', detail)}")
    else:
        st.error(str(detail))


if __name__ == "__main__":
    main()
