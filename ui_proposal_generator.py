import anthropic
import base64
import openai
import requests
import streamlit as st
from typing import Dict

st.title("UI Proposal Generator")

OPENAI_API_KEY = st.sidebar.text_input("OpenAI API Key")
GOOGLE_GEMINI_API_KEY = st.sidebar.text_input("Google Gemini API Key")
ANTHROPIC_API_KEY = st.sidebar.text_input("Anthropic API Key")

OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
GOOGLE_GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro-vision:generateContent"
ANTHROPIC_API_URL = "https://api.anthropic.com/v1/complete"

openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

DESCRIPTION_PROMPT = """
Согласно прилагаемому изображению, подробно опиши какие элементы есть на экране приложения.
Вне зависимости от языка интерфейса, отвечай на русском.
Не выводи ничего другого, кроме описания экрана.
"""

PROPOSAL_PROMPT = """
Согласно прилагаемому изображению, предложи улучшения пользовательского опыта (UX).
Выведи предложения, пронумеровав их.
Вне зависимости от языка интерфейса, отвечай на русском.
Не выводи ничего другого, кроме вышеобозначенных предложений.
"""

COMBINATION_PROMPT_TMPL = """
Три агента предложили несколько шагов по улучшению пользовательского опыта (UX), их материалы приведены ниже.
Объедини их ответы.
Вне зависимости от языка интерфейса, отвечай на русском.
Не выводи ничего другого, кроме финальных предложений.

-----
{openai_propositions}
-----
{google_gemini_propositions}
-----
{anthropic_propositions}
-----
"""


def send_dalle_request(base64_image: str, prompt: str) -> openai.ChatCompletion:
    return openai_client.chat.completions.create(
        model="dall-e-3",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        },
                    },
                ],
            },
        ]
    )


def send_openai_request(base64_image: str, prompt: str) -> openai.ChatCompletion:
    return openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        },
                    },
                ],
            },
        ]
    )


def send_google_gemini_request(base64_image: str, prompt: str) -> Dict:
    params = {
        "key": GOOGLE_GEMINI_API_KEY,
    }
    headers = {
        'Content-Type': 'application/json'
    }
    data = {
        "contents": [
            {
                "parts": [
                    {
                        "text": prompt,
                    },
                    {
                        "inline_data": {
                            "mime_type": "image/jpeg",
                            "data": base64_image,
                        },
                    },
                ],
            },
        ],
    }
    response = requests.post(GOOGLE_GEMINI_API_URL, params=params, headers=headers, json=data)
    return response.json()


def send_anthropic_request(base64_image: str, prompt: str) -> anthropic.types.Message:
    return anthropic_client.messages.create(
        model="claude-3-opus-20240229",
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                    },
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": base64_image,
                        },
                    },
                ],
            },
        ],
    )


def generate_proposals(image_bytes: bytes) -> None:
    base64_image = base64.b64encode(image_bytes).decode('utf-8')
    dalle_response = send_dalle_request(base64_image, DESCRIPTION_PROMPT)
    openai_response = send_openai_request(base64_image, PROPOSAL_PROMPT)
    openai_propositions = openai_response.choices[0].message.content
    st.text("Предложения от OpenAI:")
    st.text(openai_propositions)
    st.text("-----")
    google_gemini_response = send_google_gemini_request(base64_image, PROPOSAL_PROMPT)
    google_gemini_propositions = google_gemini_response['candidates'][0]['content']['parts'][0]['text']
    st.text("Предложения от Google Gemini:")
    st.text(google_gemini_propositions)
    st.text("-----")
    anthropic_response = send_anthropic_request(base64_image, PROPOSAL_PROMPT)
    anthropic_propositions = anthropic_response.content[0].text
    st.text("Предложения от Anthropic:")
    st.text(anthropic_propositions)
    st.text("-----")
    combination_prompt = COMBINATION_PROMPT_TMPL.format(
    openai_propositions=openai_propositions,
    google_gemini_propositions=google_gemini_propositions,
    anthropic_propositions=anthropic_propositions)
    combined_response = send_openai_request(base64_image, combination_prompt)
    combined_propositions = combined_response.choices[0].message.content
    st.text("Комбинированные предложения:")
    st.text(combined_propositions)
    st.text("-----")
    with st.form("ui_proposal_generator"):
        uploaded_file = st.file_uploader("Выберите экран UI")
        submitted = st.form_submit_button("Сгенерировать предложения")
    if uploaded_file is None:
        st.warning("Прежде чем начать генерировать преждложения необходимо загрузить экран UI!", icon="⚠")

    if uploaded_file is not None and submitted is True:
        image_bytes = uploaded_file.getvalue()
        generate_proposals(image_bytes)
