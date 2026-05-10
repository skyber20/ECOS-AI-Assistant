import streamlit as st
import time
import re
import io
import sys
from typing import List, Dict
import traceback
import requests

API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="ECOS AI Assistant",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 ECOS AI Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "is_generating" not in st.session_state:
    st.session_state.is_generating = False
if "code_outputs" not in st.session_state:
    st.session_state.code_outputs = {}

def extract_code_blocks(text: str) -> List[Dict]:
    """Извлекает блоки кода, помеченные >>>"""
    code_blocks = []
    pattern = r'>>>(.*?)>>>' ## can change on ``` 
    matches = re.finditer(pattern, text, re.DOTALL)
    
    for match in matches:
        code = match.group(1).strip()
        start = match.start()
        end = match.end()
        code_blocks.append({
            'code': code,
            'start': start,
            'end': end
        })
    
    return code_blocks

def setup_responsive_plot(fig=None, width=10, height=6):
    """Настраивает график для адаптивного отображения в Streamlit"""
    if fig is None:
        fig = plt.gcf()
    
    fig.set_size_inches(width, height)
    fig.tight_layout()
    fig.set_dpi(100)
    
    return fig

def execute_python_code(code: str) -> tuple:
    old_stdout = sys.stdout
    redirected_output = sys.stdout = io.StringIO()
    
    try:
        namespace = {}
        exec(code, namespace)
        output = redirected_output.getvalue()
        
        import matplotlib.pyplot as plt
        fig = plt.gcf()
        
        if fig.get_axes():
            fig.set_size_inches(10, 6)
            fig.tight_layout()
            return 'plot', fig
        else:
            return 'text', output if output else "Код выполнен успешно (нет вывода)"
            
    except Exception as e:
        return 'error', f"Ошибка: {str(e)}\n{traceback.format_exc()}"
    finally:
        sys.stdout = old_stdout



def simulate_llm_response(prompt: str) -> str:
    response = requests.post(
        f"{API_URL}/api/agent/run",
        json={"prompt": prompt}
    )
    return response.json()["response"]
# def simulate_llm_response(prompt: str) -> str:
#     """Имитация ответа LLM. Потом заменим на нормальную"""
#     time.sleep(0.5)
    
#     if "график" in prompt.lower() or "plot" in prompt.lower():
#         return """Вот пример кода для построения графика:

# >>>import matplotlib.pyplot as plt
# import numpy as np

# x = np.linspace(0, 10, 100)
# y = np.sin(x)

# plt.figure(figsize=(10, 6))
# plt.plot(x, y, 'b-', linewidth=2, label='sin(x)')
# plt.grid(True, alpha=0.3)
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.title('Пример графика')
# plt.legend()
# plt.show()>>>

# А вот еще один пример:

# >>>import pandas as pd
# import matplotlib.pyplot as plt

# data = {'Месяц': ['Янв', 'Фев', 'Мар', 'Апр'],
#         'Продажи': [100, 150, 130, 180]}
# df = pd.DataFrame(data)

# fig, ax = plt.subplots(figsize=(8, 5))
# ax.bar(df['Месяц'], df['Продажи'], color='skyblue')
# ax.set_title('Продажи по месяцам')
# ax.set_ylabel('Сумма продаж')
# plt.show()>>>"""
    
#     elif "таблица" in prompt.lower() or "table" in prompt.lower():
#         return """Вот пример создания таблицы:

# >>>import pandas as pd

# data = {
#     'Имя': ['Анна', 'Борис', 'Виктор', 'Галина'],
#     'Возраст': [25, 30, 35, 28],
#     'Город': ['Москва', 'СПб', 'Казань', 'Новосибирск']
# }
# df = pd.DataFrame(data)
# print(df.to_string())>>>

# И еще пример с вычислениями:

# >>>import pandas as pd
# import numpy as np

# df = pd.DataFrame({
#     'A': np.random.randn(10),
#     'B': np.random.randn(10)
# })
# df['Сумма'] = df['A'] + df['B']
# print("Таблица случайных данных:")
# print(df.round(3).to_string())
# print(f"\nСтатистика:")
# print(df.describe().round(3).to_string())>>>"""
    
#     else:
#         return f"Это ответ ассистента на ваше сообщение: '{prompt}'. (Здесь будет реальный ответ от LLM)"

def send_message():
    if st.session_state.user_input and not st.session_state.is_generating:
        st.session_state.messages.append({
            "role": "user",
            "content": st.session_state.user_input
        })
        st.session_state.is_generating = True
        st.session_state.user_input = ""

def stop_generation():
    st.session_state.is_generating = False
    st.rerun()

def get_llm_response():
    if st.session_state.is_generating:
        last_user_message = next(
            (msg["content"] for msg in reversed(st.session_state.messages) 
             if msg["role"] == "user"), 
            None
        )
        
        if last_user_message:
            response = simulate_llm_response(last_user_message)
            st.session_state.messages.append({
                "role": "assistant",
                "content": response
            })
        
        st.session_state.is_generating = False
        st.rerun()

def run_code(code_id: str, code: str):
    """Сохраняет результат выполнения кода в session_state"""
    result_type, result = execute_python_code(code)
    st.session_state.code_outputs[code_id] = {
        'type': result_type,
        'result': result ## убрать к ебеной матери эту функцию  
    }

chat_container = st.container()

with chat_container:
    for idx, message in enumerate(st.session_state.messages):
        if message["role"] == "user":
            with st.chat_message("user"):
                st.write(message["content"])
        else:
            with st.chat_message("assistant"):
                code_blocks = extract_code_blocks(message["content"])
                
                if code_blocks:
                    last_pos = 0
                    for block_idx, block in enumerate(code_blocks):
                        if block['start'] > last_pos:
                            text_part = message["content"][last_pos:block['start']]
                            if text_part.strip():
                                st.write(text_part.strip())

                        code_id = f"code_{idx}_{block_idx}"
                        

                        with st.container():
                            st.code(block['code'], language='python')
                            col1, col2 = st.columns([1, 4])
                            with col1:
                                if st.button("▶️ Запустить", key=f"run_{code_id}"):
                                    run_code(code_id, block['code'])
                                    st.rerun()
                            

                            if code_id in st.session_state.code_outputs:
                                output = st.session_state.code_outputs[code_id]
                                with st.expander("📊 Результат выполнения", expanded=True):
                                    if output['type'] == 'plot':
                                        st.pyplot(output['result'])
                                    elif output['type'] == 'text':
                                        st.text(output['result'])
                                    elif output['type'] == 'error':
                                        st.error(output['result'])
                        
                        last_pos = block['end']
                    
                    if last_pos < len(message["content"]):
                        remaining_text = message["content"][last_pos:]
                        if remaining_text.strip():
                            st.write(remaining_text.strip())
                else:
                    st.write(message["content"])
    
    if st.session_state.is_generating:
        with st.chat_message("assistant"):
            st.write("✍️ Печатает...")

with st.container():
    col1, col2, col3 = st.columns([4, 1, 1])
    
    with col1:
        st.text_input(
            "Введите сообщение...",
            key="user_input",
            disabled=st.session_state.is_generating,
            on_change=send_message,
            placeholder="Напишите что-нибудь..."
        )
    
    with col2:
        st.button(
            "📤 Отправить",
            on_click=send_message,
            disabled=st.session_state.is_generating or not st.session_state.get("user_input", ""),
            use_container_width=True
        )
    
    with col3:
        st.button(
            "⏹️ Стоп",
            on_click=stop_generation,
            disabled=not st.session_state.is_generating,
            use_container_width=True,
            type="secondary"
        )

if st.session_state.is_generating:
    get_llm_response()

with st.sidebar:
    st.header("ℹ️ Информация")
    st.write("**Статус:**", "Генерация..." if st.session_state.is_generating else "Готов")
    st.write("**Сообщений в чате:**", len(st.session_state.messages))
    st.write("**Выполненных кодов:**", len(st.session_state.code_outputs))
    
    st.divider()
    
    if st.button("🗑️ Очистить все", use_container_width=True):
        st.session_state.messages = []
        st.session_state.code_outputs = {}
        st.session_state.is_generating = False
        st.rerun()
    
    st.divider()
    

st.subheader("📊 Поддерживаемые библиотеки")
st.write("• matplotlib")
st.write("• pandas")
st.write("• numpy")
st.write("• Все стандартные библиотеки Python")

st.divider()
st.caption("💡 Сейчас стоит заглушка `simulate_llm_response()`, которую надо будет изменить")

# CSS стили
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

/* Стиль для блоков кода */
.stCodeBlock {
    border: 1px solid #e0e0e0;
    border-radius: 5px;
    margin: 10px 0;
}
</style>
""", unsafe_allow_html=True)