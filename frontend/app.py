import streamlit as st
import requests
import json
import os
import pandas as pd

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="ECOS AI Assistant",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 ECOS AI Assistant")

# Состояния
if "status" not in st.session_state:
    st.session_state.status = "Готов к запросу"
if "clarification_requests" not in st.session_state:
    st.session_state.clarification_requests = []
if "result" not in st.session_state:
    st.session_state.result = None
if "is_processing" not in st.session_state:
    st.session_state.is_processing = False
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_question" not in st.session_state:
    st.session_state.current_question = None
if "clarification_answers" not in st.session_state:
    st.session_state.clarification_answers = []
if "trigger_send" not in st.session_state:
    st.session_state.trigger_send = False
if "trigger_submit" not in st.session_state:
    st.session_state.trigger_submit = False
if "trigger_skip" not in st.session_state:
    st.session_state.trigger_skip = False
if "trigger_clear" not in st.session_state:
    st.session_state.trigger_clear = False
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "current_question_index" not in st.session_state:
    st.session_state.current_question_index = 0


def send_message():
    """Set flag to process message on next rerun"""
    if st.session_state.user_input.strip() and not st.session_state.is_processing:
        st.session_state.trigger_send = True


def submit_clarification():
    """Set flag to process clarification on next rerun"""
    if st.session_state.clarification_input.strip():
        st.session_state.trigger_submit = True


def skip_with_default():
    """Set flag to skip current question on next rerun"""
    if st.session_state.current_question and st.session_state.current_question.get("default_assumption"):
        st.session_state.trigger_skip = True


def clear_all():
    """Set flag to clear everything on next rerun"""
    st.session_state.trigger_clear = True


if st.session_state.trigger_clear:
    st.session_state.messages = []
    st.session_state.result = None
    st.session_state.status = "Готов к запросу"
    st.session_state.is_processing = False
    st.session_state.clarification_requests = []
    st.session_state.current_question = None
    st.session_state.clarification_answers = []
    st.session_state.current_question_index = 0
    st.session_state.session_id = None
    st.session_state.trigger_clear = False
    st.rerun()

if st.session_state.trigger_send:
    st.session_state.trigger_send = False
    prompt = st.session_state.user_input.strip()
    st.session_state.user_input = ""
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.status = "Анализ запроса..."
    st.session_state.is_processing = True
    st.rerun()

if st.session_state.trigger_submit:
    st.session_state.trigger_submit = False
    answer = st.session_state.clarification_input.strip()
    st.session_state.clarification_input = ""
    current = st.session_state.current_question
    st.session_state.clarification_answers.append({
        "field": current["field"],
        "question": current["question"],
        "answer": answer,
        "used_default": False
    })
    idx = st.session_state.current_question_index + 1
    if idx < len(st.session_state.clarification_requests):
        st.session_state.current_question_index = idx
        st.session_state.current_question = st.session_state.clarification_requests[idx]
        st.session_state.status = f"Уточнение {idx + 1}/{len(st.session_state.clarification_requests)}"
    else:
        st.session_state.status = "Отправка уточнений..."
        st.session_state.is_processing = True
    st.rerun()

if st.session_state.trigger_skip:
    st.session_state.trigger_skip = False
    current = st.session_state.current_question
    if current.get("default_assumption"):
        st.session_state.clarification_answers.append({
            "field": current["field"],
            "question": current["question"],
            "answer": current["default_assumption"],
            "used_default": True
        })
    idx = st.session_state.current_question_index + 1
    if idx < len(st.session_state.clarification_requests):
        st.session_state.current_question_index = idx
        st.session_state.current_question = st.session_state.clarification_requests[idx]
        st.session_state.status = f"Уточнение {idx + 1}/{len(st.session_state.clarification_requests)}"
    else:
        st.session_state.status = "Отправка уточнений..."
        st.session_state.is_processing = True
    st.rerun()


def send_clarification_answers():
    try:
        response = requests.post(
            f"{API_URL}/api/agent/clarify",
            json={
                "session_id": st.session_state.session_id,
                "answers": st.session_state.clarification_answers
            },
            timeout=60
        ).json()
        handle_api_response(response)
    except Exception as e:
        st.error(f"Ошибка соединения: {e}")
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"❌ Не удалось соединиться с сервером. ({str(e)})"
        })
        st.session_state.status = "Готов к запросу"
        st.session_state.is_processing = False


def handle_api_response(response: dict):
    if response is None:
        st.session_state.messages.append({
            "role": "assistant",
            "content": "❌ Сервер вернул пустой ответ."
        })
        st.session_state.status = "Готов к запросу"
        st.session_state.is_processing = False
        return

    status = response.get("status")
    if response.get("session_id"):
        st.session_state.session_id = response["session_id"]

    if status == "error":
        error_msg = response.get("error", "Неизвестная ошибка")
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"❌ Ошибка: {error_msg}"
        })
        st.session_state.status = "Готов к запросу"
        st.session_state.is_processing = False
        return

    if status == "design_ready":
        result = response.get("result")
        response_text = response.get("response", "Исследование завершено")
        
        # КЛЮЧЕВОЕ ИЗМЕНЕНИЕ: Сохраняем результат
        st.session_state.result = result
        
        st.session_state.messages.append({
            "role": "assistant",
            "content": response_text,
            "result": result  # Передаём результат в сообщение
        })
        st.session_state.status = "Готов к запросу"
        st.session_state.is_processing = False
        st.session_state.clarification_requests = []
        st.session_state.current_question = None
        st.session_state.clarification_answers = []

    elif status == "needs_clarification":
        clarification_requests = response.get("clarification_requests", [])
        if not clarification_requests:
            st.session_state.messages.append({
                "role": "assistant",
                "content": "ℹ️ Требуются уточнения, но вопросы не сформулированы."
            })
            st.session_state.status = "Готов к запросу"
            st.session_state.is_processing = False
            return
        
        st.session_state.clarification_requests = clarification_requests
        st.session_state.current_question_index = 0
        st.session_state.current_question = clarification_requests[0]
        st.session_state.clarification_answers = []
        st.session_state.status = f"Уточнение 1/{len(clarification_requests)}"
        st.session_state.messages.append({
            "role": "assistant",
            "content": response.get("response", "Нужны уточнения.")
        })

    else:
        result = response.get("result")
        response_text = response.get("response") or f"Статус: {status}"
        st.session_state.result = result
        st.session_state.messages.append({
            "role": "assistant",
            "content": response_text,
            "result": result
        })
        st.session_state.status = "Готов к запросу"
        st.session_state.clarification_requests = []
        st.session_state.current_question = None
        st.session_state.clarification_answers = []

    st.session_state.is_processing = False


def process_initial_request():
    last_user_message = next(
        (msg["content"] for msg in reversed(st.session_state.messages)
         if msg["role"] == "user"),
        None
    )
    if not last_user_message:
        st.session_state.is_processing = False
        return

    try:
        response = requests.post(
            f"{API_URL}/api/agent/run",
            json={
                "prompt": last_user_message,
                "session_id": st.session_state.session_id
            },
            timeout=60
        ).json()
        handle_api_response(response)
    except requests.exceptions.Timeout:
        st.session_state.messages.append({
            "role": "assistant",
            "content": "⏰ Превышено время ожидания ответа от сервера."
        })
        st.session_state.status = "Готов к запросу"
        st.session_state.is_processing = False
    except requests.exceptions.ConnectionError:
        st.session_state.messages.append({
            "role": "assistant",
            "content": "🔌 Не удалось подключиться к серверу."
        })
        st.session_state.status = "Готов к запросу"
        st.session_state.is_processing = False
    except Exception as e:
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"❌ Непредвиденная ошибка: {str(e)}"
        })
        st.session_state.status = "Готов к запросу"
        st.session_state.is_processing = False


# Process any pending actions
if st.session_state.is_processing and not st.session_state.clarification_answers:
    process_initial_request()
    st.rerun()

if st.session_state.is_processing and st.session_state.clarification_answers:
    send_clarification_answers()
    st.rerun()


def display_result_tabs(result: dict):
    """Отображает результат исследования в виде вкладок с данными"""
    if not result:
        return
    
    tabs = st.tabs(["📋 Обзор", "🏗️ Дизайн", "📊 Структура", "💻 Скрипт", "🚀 Запуск"])
    
    with tabs[0]:
        st.subheader("Обзор исследования")
        if result.get("intent"):
            intent = result["intent"]
            st.write(f"**Запрос:** {intent.get('original_query', '')}")
            st.write(f"**Тип:** {intent.get('intent_type', '')}")
            st.write(f"**Сложность:** {intent.get('complexity', '')}")
            st.write(f"**Тема:** {intent.get('topic', 'не указана')}")
            if intent.get("geography"):
                st.write(f"**География:** {', '.join(intent['geography'])}")
            if intent.get("indicators"):
                st.write(f"**Показатели:** {', '.join(intent['indicators'])}")
        
        if result.get("dataset_rerank"):
            st.subheader("Найденные источники данных")
            for source in result["dataset_rerank"].get("results", []):
                with st.expander(f"📁 {source.get('title', source.get('record_id', ''))}"):
                    st.write(f"**Релевантность:** {source.get('relevance', '')}")
                    st.write(f"**Уверенность:** {source.get('usefulness_confidence', '')}")
                    st.write(f"**Почему выбран:** {source.get('why_matched', '')}")
                    if source.get("description"):
                        st.write(f"**Описание:** {source['description']}")
                    if source.get("possible_limitations"):
                        st.warning(f"**Ограничения:** {', '.join(source['possible_limitations'])}")
    
    with tabs[1]:
        st.subheader("Дизайн исследования")
        if result.get("research_design"):
            design = result["research_design"]
            st.write(f"**Краткое описание:** {design.get('design_summary', '')}")
            
            if design.get("hypotheses"):
                st.subheader("Гипотезы")
                for h in design["hypotheses"]:
                    st.write(f"- **{h.get('id', '')}:** {h.get('statement', '')}")
            
            if design.get("required_measurements"):
                st.subheader("Необходимые измерения")
                for m in design["required_measurements"][:10]:
                    st.write(f"- **{m.get('name', '')}** (роль: {m.get('role', '')})")
            
            if design.get("visualizations"):
                st.subheader("Визуализации")
                for v in design["visualizations"]:
                    st.write(f"- {v.get('title', '')}: {v.get('chart_type', '')}")
    
    with tabs[2]:
        st.subheader("Структура целевого датасета")
        if result.get("dataset_structure"):
            structure = result["dataset_structure"]
            st.write(f"**Зернистость строки:** {structure.get('row_grain', '')}")
            st.write(f"**Первичный ключ:** {', '.join(structure.get('primary_key', []))}")
            st.write(f"**Частота:** {structure.get('expected_frequency', 'не указана')}")
            
            if structure.get("columns"):
                import pandas as pd
                df = pd.DataFrame(structure["columns"])
                st.dataframe(df[["name", "title", "role", "dtype", "unit", "nullable"]])
    
    with tabs[3]:
        st.subheader("Скрипт сборки данных")
        if result.get("build_script"):
            script = result["build_script"]
            st.write(f"**Файл:** {script.get('filename', '')}")
            st.write(f"**Запуск:** `{script.get('usage', '')}`")
            
            if script.get("inputs"):
                st.write(f"**Входы:** {', '.join(script['inputs'])}")
            
            if script.get("content"):
                st.code(script["content"], language="sql")
            
            if script.get("possible_limitations"):
                st.warning("**Ограничения скрипта:**")
                for lim in script["possible_limitations"]:
                    st.write(f"- {lim}")
    
    with tabs[4]:
        st.subheader("Результаты выполнения")
        if result.get("build_run"):
            build_run = result["build_run"]
            status = build_run.get("status", "unknown")
            if status == "succeeded":
                st.success("✅ Скрипт выполнен успешно")
            elif status == "failed":
                st.error(f"❌ Ошибка выполнения: {build_run.get('final_error', '')}")
            
            st.write(f"**Попыток:** {build_run.get('attempts_count', 0)}")
            
            output = build_run.get("output", {})
            if output:
                st.write(f"**Количество строк:** {output.get('row_count', 'N/A')}")
                
                if output.get("dataset"):
                    st.write(f"**Датасет:** `{output['dataset']}`")
                    # Пытаемся загрузить и показать CSV
                    try:
                        import pandas as pd
                        df = pd.read_csv(output['dataset'])
                        st.dataframe(df.head(10))
                    except:
                        st.info("Не удалось загрузить датасет для предпросмотра")
                
                if output.get("limitations"):
                    st.warning("**Ограничения результата:**")
                    for lim in output["limitations"]:
                        st.write(f"- {lim}")
        else:
            st.info("Скрипт ещё не выполнялся")


# Chat display
chat_container = st.container()

with chat_container:
    for message in st.session_state.messages:
        if message["role"] == "user":
            with st.chat_message("user"):
                st.write(message["content"])
        else:
            with st.chat_message("assistant"):
                st.write(message["content"])
                if message.get("result"):
                    display_result_tabs(message["result"])

    if st.session_state.is_processing and not st.session_state.clarification_answers:
        with st.chat_message("assistant"):
            st.write("✍️ Анализирую запрос...")

st.divider()

# Clarification UI
if st.session_state.current_question is not None:
    st.info(f"**{st.session_state.status}**")
    question = st.session_state.current_question
    
    with st.container():
        st.markdown(f"### ❓ {question['question']}")
        st.caption(f"📌 Поле: **{question['field']}** | 💭 Причина: {question['reason']}")
        if question.get("default_assumption"):
            st.caption(f"🔧 Значение по умолчанию: _{question['default_assumption']}_")
        
        st.markdown("---")
        col1, col2, col3 = st.columns([5, 1, 1])
        with col1:
            st.text_input(
                "Ваш ответ:",
                key="clarification_input",
                placeholder="Введите ответ или нажмите «Пропустить»",
                label_visibility="collapsed"
            )
        with col2:
            st.button(
                "📤 Ответить",
                on_click=submit_clarification,
                use_container_width=True,
                disabled=not st.session_state.get("clarification_input", "")
            )
        with col3:
            st.button(
                "⏭️ Пропустить",
                on_click=skip_with_default,
                use_container_width=True,
                disabled=not question.get("default_assumption"),
                type="secondary"
            )

elif not st.session_state.is_processing:
    with st.container():
        col1, col2 = st.columns([5, 1])
        with col1:
            st.text_input(
                "Введите сообщение...",
                key="user_input",
                placeholder="Напишите исследовательский запрос...",
                label_visibility="collapsed"
            )
        with col2:
            st.button(
                "📤 Отправить",
                on_click=send_message,
                disabled=st.session_state.is_processing or not st.session_state.get("user_input", ""),
                use_container_width=True
            )

# Sidebar
with st.sidebar:
    st.header("ℹ️ Информация")
    st.write("**Статус:**", st.session_state.status)
    st.write("**Сообщений:**", len(st.session_state.messages))
    if st.session_state.session_id:
        st.write("**Сессия:**", st.session_state.session_id[:8] + "...")

    st.divider()
    if st.button("🗑️ Очистить всё", use_container_width=True, on_click=clear_all):
        pass

    st.divider()
    st.subheader("💡 Возможности агента")
    st.write("- 🎯 Парсинг исследовательского запроса")
    st.write("- 📚 Поиск релевантных датасетов (RAG)")
    st.write("- 🏗️ Дизайн количественного исследования")
    st.write("- 📊 Проектирование структуры датасета")
    st.write("- 💻 Генерация DuckDB SQL скрипта")
    st.write("- 🚀 Автоматический запуск сборки данных")