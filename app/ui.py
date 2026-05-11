from __future__ import annotations

from io import BytesIO
import json
import mimetypes
import os
import re
from typing import Any

import altair as alt
import pandas as pd
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

    report_tab, dataset_tab, visuals_tab, artifacts_tab, json_tab = st.tabs(
        ["Отчет", "Датасет", "Визуализации", "Артефакты", "JSON"]
    )
    with report_tab:
        report = response.get("report_markdown")
        if report:
            st.markdown(report)
        else:
            st.info("Отчет пока не сформирован.")
    with dataset_tab:
        _render_dataset(response, settings["backend_url"])
    with visuals_tab:
        _render_visualizations(response, settings["backend_url"])
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

        submit_answers = st.form_submit_button("Продолжить с ответами", type="primary")
        submit_defaults = st.form_submit_button("Принять default-допущения")

    if submit_answers:
        if missing_fields:
            st.warning("Заполните все уточнения перед продолжением.")
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


def _render_dataset(response: dict[str, Any], backend_url: str) -> None:
    result = response.get("result", {})
    _render_selected_datasets(result.get("dataset_rerank") or {})

    dataset_df = _dataset_dataframe(response, backend_url)
    if dataset_df is None:
        st.info("Собранный CSV-датасет пока не найден.")
    else:
        st.markdown("### Собранный датасет")
        cols = st.columns(3)
        cols[0].metric("Строки", f"{len(dataset_df):,}".replace(",", " "))
        cols[1].metric("Колонки", str(len(dataset_df.columns)))
        cols[2].metric("Пустые значения", f"{int(dataset_df.isna().sum().sum()):,}".replace(",", " "))
        st.dataframe(dataset_df, use_container_width=True, height=360)

        dataset_artifact = _first_artifact(
            response,
            ["build_output_dataset", "build_output_target_dataset", "build_output_csv_path"],
        )
        if dataset_artifact:
            st.download_button(
                "Скачать CSV",
                data=_download_artifact(backend_url, dataset_artifact["download_url"]),
                file_name=dataset_artifact.get("filename", "target_dataset.csv"),
                mime="text/csv",
                key=f"dataset_download_{response.get('run_id')}",
            )

    metadata = _json_artifact(
        response,
        backend_url,
        ["build_output_metadata", "build_output_meta_path"],
    )
    if metadata:
        with st.expander("Metadata собранного датасета"):
            st.json(metadata, expanded=False)


def _render_selected_datasets(rerank: dict[str, Any]) -> None:
    st.markdown("### Выбранные источники")
    results = rerank.get("results") or []
    if not results:
        reason = rerank.get("no_results_reason") or "RAG не вернул выбранные источники."
        st.info(reason)
        return

    for item in results:
        with st.container(border=True):
            top_cols = st.columns([3, 1, 1])
            top_cols[0].markdown(f"**{item.get('title') or item.get('record_id')}**")
            top_cols[1].metric("Релевантность", _level_label(item.get("relevance")))
            top_cols[2].metric("Уверенность", _level_label(item.get("usefulness_confidence")))
            details = [
                ("record_id", item.get("record_id")),
                ("dataset_id", item.get("dataset_id")),
                ("source", item.get("source")),
                ("frequency", item.get("frequency")),
                ("unit", item.get("unit")),
            ]
            st.caption(" · ".join(f"{key}: {value}" for key, value in details if value))
            if item.get("description"):
                st.write(item["description"])
            if item.get("why_matched"):
                st.success(item["why_matched"])
            if item.get("possible_limitations"):
                st.warning("; ".join(str(value) for value in item["possible_limitations"]))
            if item.get("data_path") or item.get("source_url"):
                link_parts = []
                if item.get("data_path"):
                    link_parts.append(f"`{item['data_path']}`")
                if item.get("source_url"):
                    link_parts.append(f"[source_url]({item['source_url']})")
                st.markdown(" · ".join(link_parts))

    rejected = rerank.get("rejected_similar_candidates") or []
    if rejected:
        with st.expander("Похожие, но отклоненные источники"):
            for item in rejected[:12]:
                st.markdown(f"- **{item.get('title') or item.get('record_id')}**: {item.get('reason')}")


def _render_visualizations(response: dict[str, Any], backend_url: str) -> None:
    result = response.get("result", {})
    design = result.get("research_design") or {}
    hypotheses = design.get("hypotheses") or []
    visualizations = design.get("visualizations") or []

    st.markdown("### Гипотезы")
    if hypotheses:
        for item in hypotheses:
            with st.container(border=True):
                st.markdown(f"**{item.get('id') or 'H'}: {item.get('statement')}**")
                if item.get("null_hypothesis"):
                    st.caption(f"H0: {item['null_hypothesis']}")
                if item.get("expected_effect"):
                    st.write(item["expected_effect"])
    else:
        st.info("Гипотезы в дизайне не заданы.")

    st.markdown("### Графики")
    if not visualizations:
        st.info("Спецификации визуализаций в дизайне не заданы.")
        return

    dataset_df = _dataset_dataframe(response, backend_url)
    chart_df = _chart_dataframe(response, backend_url)
    structure = result.get("dataset_structure") or {}

    for spec in visualizations:
        with st.container(border=True):
            st.markdown(f"**{spec.get('title') or spec.get('id') or 'График'}**")
            meta = [
                _chart_type_label(spec.get("chart_type")),
                f"H: {spec.get('hypothesis_id')}" if spec.get("hypothesis_id") else None,
                f"x: {spec.get('x_axis')}" if spec.get("x_axis") else None,
                f"y: {spec.get('y_axis')}" if spec.get("y_axis") else None,
                f"group: {spec.get('grouping')}" if spec.get("grouping") else None,
            ]
            st.caption(" · ".join(item for item in meta if item))
            if spec.get("interpretation_guide"):
                st.info(spec["interpretation_guide"])

            chart = None
            if dataset_df is not None:
                chart = _chart_from_dataset(dataset_df, spec, structure)
            if chart is None and chart_df is not None:
                chart = _chart_from_chart_data(chart_df, spec)

            if chart is None:
                st.warning("Для этого графика пока нет подходящих числовых данных.")
            else:
                st.altair_chart(chart, use_container_width=True)


def _dataset_dataframe(response: dict[str, Any], backend_url: str) -> pd.DataFrame | None:
    return _csv_artifact_dataframe(
        response,
        backend_url,
        ["build_output_dataset", "build_output_target_dataset", "build_output_csv_path"],
    )


def _chart_dataframe(response: dict[str, Any], backend_url: str) -> pd.DataFrame | None:
    return _csv_artifact_dataframe(
        response,
        backend_url,
        [
            "build_output_chart_data",
            "build_output_chart",
            "build_output_chart_path",
            "build_output_chart_data_path",
        ],
    )


def _csv_artifact_dataframe(
    response: dict[str, Any],
    backend_url: str,
    artifact_types: list[str],
) -> pd.DataFrame | None:
    artifact = _first_artifact(response, artifact_types)
    if not artifact:
        return None
    try:
        data = _download_artifact(backend_url, artifact["download_url"])
        dataframe = pd.read_csv(BytesIO(data))
    except (requests.RequestException, pd.errors.EmptyDataError, UnicodeDecodeError) as exc:
        st.error(f"{artifact.get('filename', 'CSV')}: {exc}")
        return None
    return dataframe


def _json_artifact(
    response: dict[str, Any],
    backend_url: str,
    artifact_types: list[str],
) -> dict[str, Any] | list[Any] | None:
    artifact = _first_artifact(response, artifact_types)
    if not artifact:
        return None
    try:
        data = _download_artifact(backend_url, artifact["download_url"])
        return json.loads(data.decode("utf-8"))
    except (requests.RequestException, UnicodeDecodeError, json.JSONDecodeError):
        return None


def _first_artifact(response: dict[str, Any], artifact_types: list[str]) -> dict[str, Any] | None:
    wanted = set(artifact_types)
    for artifact in response.get("artifacts") or []:
        if artifact.get("artifact_type") in wanted and artifact.get("download_url"):
            return artifact
    return None


def _chart_from_dataset(
    dataframe: pd.DataFrame,
    spec: dict[str, Any],
    structure: dict[str, Any],
) -> alt.Chart | None:
    if dataframe.empty:
        return None

    x_column = _resolve_column(
        dataframe,
        structure,
        spec.get("x_axis"),
        fallback_names=["year", "date", "period", "time"],
    )
    y_column = _resolve_column(
        dataframe,
        structure,
        spec.get("y_axis"),
        preferred_roles=["indicator", "derived_metric"],
    )
    color_column = _resolve_column(
        dataframe,
        structure,
        spec.get("grouping"),
        fallback_names=["geo", "country", "region", "source_dataset_id"],
    )
    if not x_column or not y_column:
        return None

    plot_df = dataframe.copy()
    plot_df[y_column] = pd.to_numeric(plot_df[y_column], errors="coerce")
    plot_df = plot_df.dropna(subset=[x_column, y_column])
    if plot_df.empty:
        return None

    chart_type = str(spec.get("chart_type") or "").lower()
    base = alt.Chart(plot_df).encode(
        x=_x_encoding(plot_df, x_column),
        y=alt.Y(f"{y_column}:Q", title=spec.get("y_axis") or y_column),
        tooltip=[column for column in [x_column, y_column, color_column] if column],
    )
    if color_column and plot_df[color_column].nunique(dropna=True) > 1:
        base = base.encode(color=alt.Color(f"{color_column}:N", title=color_column))

    if "bar" in chart_type or "column" in chart_type:
        return base.mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3).properties(height=340)
    if "scatter" in chart_type or "point" in chart_type:
        return base.mark_circle(size=80, opacity=0.78).properties(height=340)
    if "area" in chart_type:
        return base.mark_area(opacity=0.55, line=True).properties(height=340)
    return base.mark_line(point=True, strokeWidth=3).properties(height=340)


def _chart_from_chart_data(chart_dataframe: pd.DataFrame, spec: dict[str, Any]) -> alt.Chart | None:
    if chart_dataframe.empty or "value" not in chart_dataframe.columns:
        return None

    plot_df = chart_dataframe.copy()
    spec_id = spec.get("id")
    title = spec.get("title")
    if spec_id and "visualization_id" in plot_df.columns:
        filtered = plot_df[plot_df["visualization_id"].astype(str) == str(spec_id)]
        if not filtered.empty:
            plot_df = filtered
    elif title and "title" in plot_df.columns:
        filtered = plot_df[plot_df["title"].astype(str) == str(title)]
        if not filtered.empty:
            plot_df = filtered

    x_column = "year" if "year" in plot_df.columns else _first_existing_column(plot_df, ["date", "period", "x"])
    y_column = "value"
    color_column = _first_existing_column(plot_df, ["geo", "metric"])
    if not x_column:
        return None

    plot_df[y_column] = pd.to_numeric(plot_df[y_column], errors="coerce")
    plot_df = plot_df.dropna(subset=[x_column, y_column])
    if plot_df.empty:
        return None

    chart_type = str(spec.get("chart_type") or "").lower()
    base = alt.Chart(plot_df).encode(
        x=_x_encoding(plot_df, x_column),
        y=alt.Y(f"{y_column}:Q", title=spec.get("y_axis") or "value"),
        tooltip=[column for column in [x_column, y_column, color_column, "metric"] if column in plot_df.columns],
    )
    if color_column and plot_df[color_column].nunique(dropna=True) > 1:
        base = base.encode(color=alt.Color(f"{color_column}:N", title=color_column))
    if "bar" in chart_type or "column" in chart_type:
        return base.mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3).properties(height=340)
    if "scatter" in chart_type or "point" in chart_type:
        return base.mark_circle(size=80, opacity=0.78).properties(height=340)
    return base.mark_line(point=True, strokeWidth=3).properties(height=340)


def _resolve_column(
    dataframe: pd.DataFrame,
    structure: dict[str, Any],
    requested: Any,
    preferred_roles: list[str] | None = None,
    fallback_names: list[str] | None = None,
) -> str | None:
    columns = list(dataframe.columns)
    candidates: list[Any] = [requested]
    if fallback_names:
        candidates.extend(fallback_names)

    for candidate in candidates:
        column = _match_column(columns, candidate)
        if column:
            return column

    structured_columns = structure.get("columns") if isinstance(structure, dict) else []
    if preferred_roles and isinstance(structured_columns, list):
        for role in preferred_roles:
            for item in structured_columns:
                if not isinstance(item, dict) or item.get("role") != role:
                    continue
                column = _match_column(columns, item.get("name")) or _match_column(columns, item.get("title"))
                if column:
                    return column

    numeric_columns = []
    dimension_keys = {"year", "date", "period", "time", "geo", "country", "region"}
    for column in columns:
        if _normalize_key(column) in dimension_keys:
            continue
        numeric = pd.to_numeric(dataframe[column], errors="coerce")
        if numeric.notna().mean() > 0.5:
            numeric_columns.append(column)
    return numeric_columns[0] if numeric_columns else None


def _match_column(columns: list[str], value: Any) -> str | None:
    if value is None:
        return None
    wanted = _normalize_key(value)
    if not wanted:
        return None
    for column in columns:
        if _normalize_key(column) == wanted:
            return column
    for column in columns:
        key = _normalize_key(column)
        if wanted in key or key in wanted:
            return column
    return None


def _x_encoding(dataframe: pd.DataFrame, column: str) -> alt.X:
    numeric = pd.to_numeric(dataframe[column], errors="coerce")
    if numeric.notna().mean() > 0.8:
        values = list(dataframe[column].dropna().unique())
        try:
            sort_values = sorted(values)
        except TypeError:
            sort_values = sorted(values, key=lambda value: str(value))
        return alt.X(f"{column}:O", sort=sort_values, title=column)
    return alt.X(f"{column}:N", sort=None, title=column)


def _first_existing_column(dataframe: pd.DataFrame, names: list[str]) -> str | None:
    for name in names:
        column = _match_column(list(dataframe.columns), name)
        if column:
            return column
    return None


def _normalize_key(value: Any) -> str:
    return re.sub(r"[^a-zа-я0-9]+", "", str(value).lower())


def _level_label(value: Any) -> str:
    labels = {
        "high": "Высокая",
        "medium": "Средняя",
        "low": "Низкая",
    }
    return labels.get(str(value or "").lower(), str(value or "—"))


def _chart_type_label(value: Any) -> str:
    labels = {
        "line": "Линия",
        "bar": "Столбцы",
        "column": "Столбцы",
        "scatter": "Точки",
        "area": "Площадь",
        "heatmap": "Теплокарта",
    }
    text = str(value or "").lower()
    return labels.get(text, str(value or "График"))


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
