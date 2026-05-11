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
    "no_data": "Нет данных",
    "unsupported": "Не поддерживается",
}


def main() -> None:
    st.set_page_config(
        page_title="ECOS AI Assistant",
        layout="wide",
    )
    _apply_style()

    st.title("ECOS AI Assistant")

    settings = _sidebar_settings()
    _render_health(settings["backend_url"])

    with st.form("research_form", clear_on_submit=False):
        query = st.text_area(
            "Запрос",
            value=st.session_state.get("query", ""),
            height=120,
            placeholder="Например: Исследуй секрет рецепта маленькой инфляции в США",
        )
        submitted = st.form_submit_button("Запустить исследование", type="primary")

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

    if not health.get("catalog_records_exists"):
        st.warning("Каталог `data/catalog_records.jsonl` не найден.")
    if not health.get("dumps_exists"):
        st.info("Папка `dumps/` не найдена: сборка датасета может вернуть только metadata.")


def _start_research(query: str, settings: dict[str, Any]) -> None:
    if not query.strip():
        st.warning("Запрос пустой.")
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

    cols = st.columns([1, 2, 2])
    cols[0].metric("Статус", STATUS_LABELS.get(status, status))
    cols[1].text_input("Run ID", value=response.get("run_id", ""), disabled=True)
    cols[2].text_input("Артефакты", value=response.get("artifact_dir", ""), disabled=True)

    message = response.get("message")
    if message:
        st.info(message)

    if status == "needs_clarification":
        _render_clarifications(response, settings)

    report_tab, artifacts_tab, json_tab = st.tabs(["Отчет", "Артефакты", "JSON"])
    with report_tab:
        report = response.get("report_markdown")
        if report:
            st.markdown(report)
        else:
            st.info("Отчет пока не сформирован.")
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
        for index, item in enumerate(requests_payload, start=1):
            default = item.get("default_assumption") or ""
            answer = st.text_area(
                item.get("question") or f"Уточнение {index}",
                value=default,
                help=item.get("reason") or None,
                key=f"clarification_{response.get('run_id')}_{index}",
            )
            used_default = bool(default and answer.strip() == default.strip())
            if not answer.strip():
                missing_fields.append(item.get("field") or str(index))
            answers.append(
                {
                    "field": item.get("field") or f"clarification_{index}",
                    "question": item.get("question") or "",
                    "answer": answer.strip(),
                    "used_default": used_default,
                }
            )

        submitted = st.form_submit_button("Продолжить", type="primary")

    if submitted:
        if missing_fields:
            st.warning("Заполните все уточнения перед продолжением.")
            return
        _continue_research(response, settings, answers)


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
    if exc.response is not None:
        try:
            payload = exc.response.json()
            detail = payload.get("detail", payload)
        except json.JSONDecodeError:
            detail = exc.response.text
    if isinstance(detail, dict):
        st.error(f"{detail.get('type', 'Ошибка')}: {detail.get('message', detail)}")
    else:
        st.error(str(detail))


def _apply_style() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1180px;
        }
        .stMetric {
            border: 1px solid rgba(49, 51, 63, 0.14);
            border-radius: 8px;
            padding: 0.65rem 0.8rem;
        }
        .stDownloadButton button {
            width: 100%;
            justify-content: flex-start;
        }
        textarea {
            font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
