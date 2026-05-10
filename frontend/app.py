import streamlit as st
import requests
import json

API_URL = "http://localhost:8000"

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


# Process triggers in the main flow (not in callbacks)
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

    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })
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

    # Переход к следующему вопросу
    idx = st.session_state.current_question_index + 1
    if idx < len(st.session_state.clarification_requests):
        st.session_state.current_question_index = idx
        st.session_state.current_question = st.session_state.clarification_requests[idx]
        st.session_state.status = f"Уточнение {idx + 1}/{len(st.session_state.clarification_requests)}"
    else:
        # Отправляем ответы на бекенд
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
            timeout=30
        ).json()

        handle_api_response(response)
    except Exception as e:
        st.error(f"Ошибка соединения: {e}")
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"❌ Не удалось соединиться с сервером. Попробуйте ещё раз. ({str(e)})"
        })
        st.session_state.status = "Ошибка"
        st.session_state.is_processing = False


def handle_api_response(response: dict):
    # Проверка на null/None ответ
    if response is None:
        st.session_state.messages.append({
            "role": "assistant",
            "content": "❌ Сервер вернул пустой ответ. Возможно, произошла внутренняя ошибка. Попробуйте переформулировать запрос."
        })
        st.session_state.status = "Готов к запросу"
        st.session_state.is_processing = False
        st.session_state.clarification_requests = []
        st.session_state.current_question = None
        st.session_state.clarification_answers = []
        return

    status = response.get("status")
    
    # Сохраняем session_id если пришел
    if response.get("session_id"):
        st.session_state.session_id = response["session_id"]

    # Обработка ошибок
    if status == "error":
        error_msg = response.get("error", "Неизвестная ошибка")
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"❌ Произошла ошибка: {error_msg}"
        })
        st.session_state.status = "Готов к запросу"
        st.session_state.is_processing = False
        st.session_state.clarification_requests = []
        st.session_state.current_question = None
        st.session_state.clarification_answers = []
        return

    # Обработка design_ready
    if status == "design_ready":
        result = response.get("result")
        if result is None:
            # Если result = null, но статус design_ready
            st.session_state.messages.append({
                "role": "assistant",
                "content": "✅ Дизайн исследования создан, но результат не содержит данных. Возможно, запрос требует уточнения."
            })
        else:
            st.session_state.result = result
            response_text = response.get("response", "Готово")
            st.session_state.messages.append({
                "role": "assistant",
                "content": response_text,
                "result": result
            })
        st.session_state.status = "Готов к запросу"
        st.session_state.clarification_requests = []
        st.session_state.current_question = None
        st.session_state.clarification_answers = []

    # Обработка needs_clarification
    elif status == "needs_clarification":
        clarification_requests = response.get("clarification_requests", [])
        if not clarification_requests:
            # Если запрошены уточнения, но список пуст
            st.session_state.messages.append({
                "role": "assistant",
                "content": "ℹ️ Требуются уточнения, но система не смогла сформулировать вопросы. Попробуйте описать задачу подробнее."
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

    # Обработка других статусов
    else:
        response_text = response.get("response") or response.get("error") or "Неизвестный статус"
        if response_text is None:
            response_text = "ℹ️ Получен ответ от сервера, но текст сообщения отсутствует."
        
        result = response.get("result")
        st.session_state.messages.append({
            "role": "assistant",
            "content": response_text,
            "result": result if result else None
        })
        st.session_state.status = "Готов к запросу"

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
            timeout=30
        ).json()

        handle_api_response(response)
    except requests.exceptions.Timeout:
        st.session_state.messages.append({
            "role": "assistant",
            "content": "⏰ Превышено время ожидания ответа от сервера. Попробуйте упростить запрос."
        })
        st.session_state.status = "Готов к запросу"
        st.session_state.is_processing = False
    except requests.exceptions.ConnectionError:
        st.session_state.messages.append({
            "role": "assistant",
            "content": "🔌 Не удалось подключиться к серверу. Убедитесь, что API запущен."
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


# Основная логика
if st.session_state.is_processing and not st.session_state.clarification_answers:
    process_initial_request()
    st.rerun()

if st.session_state.is_processing and st.session_state.clarification_answers:
    send_clarification_answers()
    st.rerun()

# Интерфейс
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
                    with st.expander("📋 Дизайн исследования (JSON)", expanded=True):
                        st.json(message["result"])

    if st.session_state.is_processing and not st.session_state.clarification_answers:
        with st.chat_message("assistant"):
            st.write("✍️ Анализирую запрос...")

# Поле ввода или уточнений
st.divider()

if st.session_state.current_question is not None:
    # Режим уточнений
    st.info(f"**{st.session_state.status}**")
    question = st.session_state.current_question
    with st.container():
        st.markdown(f"### ❓ {question['question']}")
        st.caption(f"Поле: {question['field']}. Причина: {question['reason']}")
        if question.get("default_assumption"):
            st.caption(f"Default: {question['default_assumption']}")

        col1, col2, col3 = st.columns([4, 1, 1])
        with col1:
            st.text_input(
                "Ваш ответ:",
                key="clarification_input",
                placeholder="Введите ответ или нажмите Пропустить для default",
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
    # Обычный режим ввода
    col1, col2 = st.columns([4, 1])
    with col1:
        st.text_input(
            "Введите сообщение...",
            key="user_input",
            placeholder="Напишите исследовательский запрос..."
        )
    with col2:
        st.button(
            "📤 Отправить",
            on_click=send_message,
            disabled=st.session_state.is_processing or not st.session_state.get("user_input", ""),
            use_container_width=True
        )


# Боковая панель
with st.sidebar:
    st.header("ℹ️ Информация")
    st.write("**Статус:**", st.session_state.status)
    st.write("**Сообщений в чате:**", len(st.session_state.messages))
    if st.session_state.session_id:
        st.write("**Сессия:**", st.session_state.session_id[:8] + "...")

    st.divider()

    if st.button("🗑️ Очистить всё", use_container_width=True, on_click=clear_all):
        pass

    st.divider()
    st.subheader("💡 Подсказка")
    st.write("Текущая версия возвращает результат 2 тула.")
    st.write("После завершения вы получите готовый дизайн исследования в JSON.")

st.markdown("""
<style>
.stTextInput > div > div > input {
    font-size: 16px;
}
.stChatMessage {
    margin-bottom: 10px;
}
.main .block-container {
    padding-top: 1rem;
    padding-bottom: 0rem;
}
</style>
""", unsafe_allow_html=True)