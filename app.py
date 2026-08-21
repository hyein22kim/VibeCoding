import json
import os
from datetime import datetime
from zoneinfo import ZoneInfo

import streamlit as st
from openai import APIConnectionError, AuthenticationError, OpenAI, RateLimitError
from pypdf import PdfReader
from supabase import Client, create_client


SUMMARY_SYSTEM_PROMPT = """
PDF 텍스트를 분석해 반드시 다음 항목을 JSON 형식으로 반환하세요.
- 핵심 요약: 정확히 3문장
- 주요 키워드: 정확히 5개
- 난이도: 상, 중, 하 중 하나
JSON 외의 설명이나 마크다운은 반환하지 마세요.
""".strip()

SUMMARY_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "핵심 요약": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 3,
            "maxItems": 3,
        },
        "주요 키워드": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 5,
            "maxItems": 5,
        },
        "난이도": {
            "type": "string",
            "enum": ["상", "중", "하"],
        },
    },
    "required": ["핵심 요약", "주요 키워드", "난이도"],
    "additionalProperties": False,
}

QUIZ_SYSTEM_PROMPT = """
당신은 학습 퀴즈를 만드는 도우미입니다.
주어진 요약을 바탕으로 4지선다 퀴즈 3문항을 만드세요.
각 문항은 question, options(4개), answer(정답 인덱스, 0부터 시작), explanation을 포함해야 합니다.
결과는 {"questions": [...]} 형태의 JSON으로 반환하세요.
"""

TUTOR_SESSIONS_TABLE = "tutor sessions"


# 현재 Streamlit 버전에 맞는 방식으로 앱을 다시 실행한다.
def rerun_app():
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()


# 앱 전체에 일관된 색상, 여백, 카드와 버튼 스타일을 적용한다.
def apply_custom_styles():
    st.markdown(
        """
        <style>
        :root {
            --app-primary: #4f46e5;
            --app-primary-soft: #eef2ff;
            --app-border: #e2e8f0;
            --app-text-soft: #64748b;
        }

        .stApp {
            background: linear-gradient(180deg, #f8fafc 0%, #ffffff 320px);
        }

        .block-container {
            max-width: 1080px;
            padding-top: 2rem;
            padding-bottom: 4rem;
        }

        [data-testid="stSidebar"] {
            background: #ffffff;
            border-right: 1px solid var(--app-border);
        }

        [data-testid="stSidebar"] .stRadio > label {
            color: var(--app-text-soft);
            font-weight: 600;
        }

        div.stButton > button,
        div.stFormSubmitButton > button,
        div.stLinkButton > a {
            border-radius: 10px;
            min-height: 2.6rem;
            font-weight: 650;
            transition: transform 0.15s ease, box-shadow 0.15s ease;
        }

        div.stButton > button:hover,
        div.stFormSubmitButton > button:hover,
        div.stLinkButton > a:hover {
            transform: translateY(-1px);
            box-shadow: 0 6px 16px rgba(79, 70, 229, 0.14);
        }

        [data-testid="stTextArea"] textarea,
        [data-testid="stFileUploaderDropzone"],
        [data-testid="stExpander"] {
            border-radius: 12px;
        }

        [data-testid="stTabs"] [data-baseweb="tab-list"] {
            gap: 0.4rem;
            border-bottom: 1px solid var(--app-border);
        }

        [data-testid="stTabs"] button[role="tab"] {
            padding: 0.75rem 1rem;
            border-radius: 10px 10px 0 0;
            font-weight: 650;
        }

        [data-testid="stTabs"] button[aria-selected="true"] {
            color: var(--app-primary);
            background: var(--app-primary-soft);
        }

        [data-testid="stChatMessage"] {
            border: 1px solid var(--app-border);
            border-radius: 14px;
            padding: 0.4rem 0.8rem;
            background: rgba(255, 255, 255, 0.86);
        }

        [data-testid="stAlert"] {
            border-radius: 12px;
        }

        hr {
            border-color: var(--app-border);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# Streamlit secrets의 접속 정보로 재사용 가능한 Supabase 클라이언트를 만든다.
@st.cache_resource
def get_supabase_client():
    try:
        supabase_url = st.secrets.get("SUPABASE_URL")
        supabase_key = st.secrets.get("SUPABASE_KEY")
    except Exception:
        supabase_url = None
        supabase_key = None

    if not supabase_url or not supabase_key:
        raise ValueError(
            "secrets.toml에 SUPABASE_URL과 SUPABASE_KEY를 설정해 주세요."
        )

    return create_client(supabase_url, supabase_key)


# Supabase memos 테이블에 메모 내용만 추가한다.
def add_memo(content):
    client: Client = get_supabase_client()
    client.table("memos").insert({"content": content}).execute()


# Supabase memos 테이블에서 메모를 최신순으로 조회한다.
def fetch_memos():
    client: Client = get_supabase_client()
    response = (
        client.table("memos")
        .select("id, content, created_at")
        .order("created_at", desc=True)
        .execute()
    )
    return response.data or []


# Supabase memos 테이블에서 지정한 ID의 메모를 삭제한다.
def delete_memo(memo_id):
    client: Client = get_supabase_client()
    client.table("memos").delete().eq("id", memo_id).execute()


# Supabase timestamptz 값을 한국 시간의 화면 표시용 문자열로 변환한다.
def format_created_at(created_at):
    if not created_at:
        return "저장 시각 없음"

    try:
        saved_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        saved_at = saved_at.astimezone(ZoneInfo("Asia/Seoul"))
        return saved_at.strftime("%Y-%m-%d %H:%M:%S")
    except (TypeError, ValueError):
        return str(created_at)


# 요약과 퀴즈를 새 튜터 세션 행에 저장하고 data[0]의 ID를 반환한다.
def create_tutor_session(summary, quiz):
    client: Client = get_supabase_client()
    response = (
        client.table(TUTOR_SESSIONS_TABLE)
        .insert(
            {
                "summary": summary,
                "quiz": quiz,
                "chat_history": [],
            }
        )
        .execute()
    )

    if not response.data:
        raise ValueError("생성된 튜터 세션 정보를 받지 못했습니다.")

    session_row = response.data[0]
    return session_row["id"]


# ID로 튜터 세션 행 하나를 조회한다.
def fetch_tutor_session(session_id):
    client: Client = get_supabase_client()
    response = (
        client.table(TUTOR_SESSIONS_TABLE)
        .select("id, summary, quiz, chat_history, created_at")
        .eq("id", session_id)
        .limit(1)
        .execute()
    )
    return response.data[0] if response.data else None


# tutor_sessions 테이블의 세션 목록을 생성 시각 최신순으로 조회한다.
def fetch_tutor_sessions():
    client: Client = get_supabase_client()
    response = (
        client.table(TUTOR_SESSIONS_TABLE)
        .select("id, summary, created_at")
        .order("created_at", desc=True)
        .execute()
    )
    return response.data or []


# 지정한 튜터 세션 행의 대화 기록을 최신 내용으로 갱신한다.
def update_tutor_chat_history(session_id, chat_history):
    client: Client = get_supabase_client()
    (
        client.table(TUTOR_SESSIONS_TABLE)
        .update({"chat_history": chat_history})
        .eq("id", session_id)
        .execute()
    )


# URL의 session_id가 가리키는 튜터 상태를 Supabase에서 복원한다.
def restore_tutor_session_from_query():
    session_id = st.query_params.get("session_id")
    if not session_id:
        return False

    if (
        str(st.session_state.get("tutor_session_id", "")) == str(session_id)
        and "pdf_summary" in st.session_state
        and "pdf_quiz" in st.session_state
    ):
        st.session_state.tutor_view = "detail"
        return True

    try:
        session_row = fetch_tutor_session(session_id)
    except Exception:
        st.error("저장된 튜터 세션을 불러오지 못했습니다.")
        return False

    if not session_row:
        st.warning("해당 튜터 세션을 찾을 수 없습니다.")
        return False

    st.session_state.pdf_summary = session_row["summary"]
    st.session_state.pdf_quiz = session_row["quiz"]
    st.session_state.tutor_messages = session_row.get("chat_history") or []
    st.session_state.quiz_answers = {}
    st.session_state.tutor_session_id = session_row["id"]
    st.session_state.restored_session_id = str(session_row["id"])
    st.session_state.tutor_view = "detail"
    return True


# 사이드바 메뉴를 표시하고 사용자가 선택한 메뉴를 반환한다.
def render_sidebar():
    default_index = 1 if st.query_params.get("session_id") else 0
    st.sidebar.markdown("### 🧭 메뉴")
    st.sidebar.caption("원하는 작업 공간을 선택하세요.")
    return st.sidebar.radio(
        "메뉴",
        ["메모", "나만의 튜터"],
        index=default_index,
        label_visibility="collapsed",
    )


# 직접 입력한 텍스트를 새 메모로 저장하는 입력 폼을 표시한다.
def render_memo_form():
    with st.form("memo_form", clear_on_submit=True):
        memo = st.text_area(
            "메모",
            placeholder="기억하고 싶은 내용을 입력하세요.",
            height=140,
        )
        saved = st.form_submit_button("저장", type="primary")

    if not saved:
        return

    content = memo.strip()
    if not content:
        st.warning("저장할 메모를 입력해 주세요.")
        return

    try:
        add_memo(content)
        rerun_app()
    except Exception:
        st.error("메모를 저장하지 못했습니다. Supabase 연결 설정을 확인해 주세요.")


# 저장된 메모를 최신순으로 표시하고 각 메모의 삭제 기능을 제공한다.
def render_memo_list():
    st.subheader("저장된 메모")

    try:
        memos = fetch_memos()
    except Exception:
        st.error("메모를 불러오지 못했습니다. Supabase 연결 설정을 확인해 주세요.")
        return

    if not memos:
        st.info("아직 저장된 메모가 없습니다.")
        return

    for memo_item in memos:
        memo_id = memo_item["id"]
        content = memo_item["content"]
        saved_at = format_created_at(memo_item.get("created_at"))
        memo_column, delete_column = st.columns([8, 1])

        with memo_column:
            if len(content) >= 50:
                st.write(f"{content[:50]}…")
                with st.expander("더보기"):
                    st.write(content)
            else:
                st.write(content)

            st.caption(f"저장 시각: {saved_at}")

        with delete_column:
            if st.button("삭제", key=f"delete_{memo_id}"):
                try:
                    delete_memo(memo_id)
                    rerun_app()
                except Exception:
                    st.error("메모를 삭제하지 못했습니다. Supabase 연결을 확인해 주세요.")

        st.divider()


# 직접 입력 화면과 저장된 메모 목록을 함께 표시한다.
def render_memo_page():
    st.header("메모")
    st.caption("생각을 빠르게 기록하고 Supabase에서 안전하게 관리하세요.")
    render_memo_form()
    render_memo_list()


# 업로드한 PDF에서 페이지 번호와 페이지별 텍스트를 추출한다.
def extract_pdf_pages(uploaded_pdf):
    reader = PdfReader(uploaded_pdf)
    return [
        (page_number, (page.extract_text() or "").strip())
        for page_number, page in enumerate(reader.pages, start=1)
    ]


# 텍스트가 있는 PDF 페이지를 페이지 제목이 포함된 메모 문자열로 합친다.
def build_pdf_memo_text(pages_with_text):
    return "\n\n".join(
        f"{page_number}페이지\n{page_text}"
        for page_number, page_text in pages_with_text
    )


# PDF의 추출 결과를 페이지별 접기 영역으로 표시한다.
def render_pdf_pages(page_texts):
    st.subheader("PDF에서 추출된 텍스트")

    for page_number, page_text in page_texts:
        with st.expander(f"{page_number}페이지"):
            if page_text:
                st.text(page_text)
            else:
                st.info("이 페이지에서 추출된 텍스트가 없습니다.")


# Streamlit secrets를 먼저 확인하고 없으면 환경변수에서 API 키를 가져온다.
def get_openai_api_key():
    try:
        secret_key = st.secrets.get("OPENAI_API_KEY")
    except Exception:
        secret_key = None

    return secret_key or os.getenv("OPENAI_API_KEY")


# OpenAI Responses API로 PDF 텍스트의 구조화된 요약을 생성한다.
def summarize_pdf_text(pdf_text):
    api_key = get_openai_api_key()
    if not api_key:
        raise ValueError(
            "Streamlit secrets 또는 OPENAI_API_KEY 환경변수를 설정해 주세요."
        )

    client = OpenAI(api_key=api_key)
    response = client.responses.create(
        model="gpt-5-mini",
        input=[
            {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"다음 PDF 텍스트를 요약해 주세요.\n\n{pdf_text}",
            },
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "pdf_summary",
                "schema": SUMMARY_JSON_SCHEMA,
                "strict": True,
            }
        },
    )
    return json.loads(response.output_text)


# 구조화된 PDF 요약을 요약문, 키워드, 난이도로 나눠 표시한다.
def render_pdf_summary(summary):
    st.subheader("AI 요약")

    st.markdown("#### 핵심 요약")
    for index, sentence in enumerate(summary["핵심 요약"], start=1):
        st.write(f"{index}. {sentence}")

    st.markdown("#### 주요 키워드")
    st.markdown(" · ".join(f"`{keyword}`" for keyword in summary["주요 키워드"]))

    st.markdown("#### 난이도")
    st.info(summary["난이도"])


# 기존 핵심 요약 3문장만 사용해 OpenAI API로 퀴즈 JSON을 생성한다.
def generate_quiz(summary):
    api_key = get_openai_api_key()
    if not api_key:
        raise ValueError(
            "Streamlit secrets 또는 OPENAI_API_KEY 환경변수를 설정해 주세요."
        )

    summary_text = "\n".join(summary["핵심 요약"])
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-5-mini",
        messages=[
            {"role": "system", "content": QUIZ_SYSTEM_PROMPT},
            {"role": "user", "content": summary_text},
        ],
        response_format={"type": "json_object"},
    )
    quiz = json.loads(response.choices[0].message.content)

    questions = quiz.get("questions")
    if not isinstance(questions, list) or len(questions) != 3:
        raise ValueError("퀴즈 응답에 3개의 문항이 없습니다.")

    for question in questions:
        options = question.get("options")
        answer = question.get("answer")
        if (
            not question.get("question")
            or not isinstance(options, list)
            or len(options) != 4
            or not isinstance(answer, int)
            or answer not in range(4)
            or not question.get("explanation")
        ):
            raise ValueError("퀴즈 문항 형식이 올바르지 않습니다.")

    return quiz


# 퀴즈를 문항별로 표시하고 선택한 답과 채점 결과를 세션에 저장한다.
def render_quiz(quiz):
    st.subheader("학습 퀴즈")

    if "quiz_answers" not in st.session_state:
        st.session_state.quiz_answers = {}

    for index, question in enumerate(quiz["questions"]):
        st.markdown(f"#### {index + 1}. {question['question']}")
        options = question["options"]
        selected_answer = st.radio(
            f"{index + 1}번 문제 보기",
            options=range(4),
            format_func=lambda option_index, choices=options: choices[option_index],
            index=None,
            key=f"quiz_choice_{index}",
            label_visibility="collapsed",
        )

        if selected_answer is not None:
            st.session_state.quiz_answers[index] = selected_answer

            if selected_answer == question["answer"]:
                st.success("정답입니다!")
            else:
                correct_option = options[question["answer"]]
                st.error(f"오답입니다. 정답은 '{correct_option}'입니다.")

            st.caption(f"해설: {question['explanation']}")

        st.divider()


# OpenAI API 예외를 사용자가 이해할 수 있는 안내 문구로 변환한다.
def get_summary_error_message(error):
    error_code = getattr(error, "code", None)
    error_body = getattr(error, "body", None)

    if not error_code and isinstance(error_body, dict):
        error_code = error_body.get("code")

    if error_code in {"credit_balance_exhausted", "insufficient_quota"}:
        return (
            "OpenAI API 크레딧이 부족합니다. 결제 설정에서 크레딧을 "
            "충전한 뒤 다시 시도해 주세요."
        )

    if isinstance(error, AuthenticationError):
        return "OpenAI API 키가 올바르지 않습니다. secrets 설정을 확인해 주세요."

    if isinstance(error, APIConnectionError):
        return "OpenAI API에 연결할 수 없습니다. 네트워크 상태를 확인해 주세요."

    if isinstance(error, RateLimitError):
        return "OpenAI API 요청 한도를 초과했습니다. 잠시 후 다시 시도해 주세요."

    return "요약을 생성하지 못했습니다. 잠시 후 다시 시도해 주세요."


# 퀴즈 생성 오류를 기존 OpenAI 오류 분류에 맞는 안내 문구로 변환한다.
def get_quiz_error_message(error):
    return get_summary_error_message(error).replace("요약을", "퀴즈를")


# OpenAI 오류 안내를 표시하고 크레딧 부족이면 결제 설정 링크를 제공한다.
def render_openai_error(error, action="요약"):
    action_text = {
        "요약": "요약을",
        "퀴즈": "퀴즈를",
        "답변": "답변을",
    }.get(action, f"{action}을")
    message = get_summary_error_message(error).replace("요약을", action_text)
    st.error(message)

    error_code = getattr(error, "code", None)
    error_body = getattr(error, "body", None)
    if not error_code and isinstance(error_body, dict):
        error_code = error_body.get("code")

    if error_code in {"credit_balance_exhausted", "insufficient_quota"}:
        st.link_button(
            "OpenAI 결제 설정 열기",
            "https://platform.openai.com/settings/organization/billing/overview",
        )


# 현재 퀴즈 답안에서 사용자가 틀린 문제의 상세 정보를 반환한다.
def get_wrong_questions(quiz):
    answers = st.session_state.get("quiz_answers", {})
    wrong_questions = []

    for index, question in enumerate(quiz["questions"]):
        if index not in answers or answers[index] == question["answer"]:
            continue

        wrong_questions.append(
            {
                "문항 번호": index + 1,
                "문제": question["question"],
                "선택한 답": question["options"][answers[index]],
                "정답": question["options"][question["answer"]],
                "해설": question["explanation"],
            }
        )

    return wrong_questions


# 요약, 전체 퀴즈, 틀린 문제를 포함한 튜터용 시스템 프롬프트를 만든다.
def build_tutor_system_prompt(summary, quiz, wrong_questions):
    return f"""
당신은 업로드된 학습 문서를 설명하는 친절한 개인 튜터입니다.
아래 요약, 전체 퀴즈, 틀린 문제를 바탕으로 사용자의 질문에 정확하게 답하세요.
문서에 없는 내용은 추측하지 말고, 필요한 경우 정보가 부족하다고 알려주세요.

[요약]
{json.dumps(summary, ensure_ascii=False)}

[퀴즈 전체]
{json.dumps(quiz, ensure_ascii=False)}

[틀린 문제]
{json.dumps(wrong_questions, ensure_ascii=False)}
""".strip()


# 저장된 대화와 학습 정보를 사용해 OpenAI API에서 튜터 답변을 생성한다.
def ask_tutor(summary, quiz, messages):
    api_key = get_openai_api_key()
    if not api_key:
        raise ValueError(
            "Streamlit secrets 또는 OPENAI_API_KEY 환경변수를 설정해 주세요."
        )

    wrong_questions = get_wrong_questions(quiz)
    system_prompt = build_tutor_system_prompt(summary, quiz, wrong_questions)
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-5-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            *messages,
        ],
    )
    return response.choices[0].message.content


# 사용자 질문을 대화에 저장하고 튜터 답변을 생성해 화면에 표시한다.
def submit_tutor_message(prompt, summary, quiz, spinner_message):
    st.session_state.tutor_messages.append(
        {"role": "user", "content": prompt}
    )

    session_id = st.session_state.get("tutor_session_id")
    if session_id:
        try:
            update_tutor_chat_history(
                session_id,
                st.session_state.tutor_messages,
            )
        except Exception:
            st.warning("질문을 Supabase 대화 기록에 저장하지 못했습니다.")

    with st.chat_message("user"):
        st.write(prompt)

    try:
        with st.spinner(spinner_message):
            answer = ask_tutor(
                summary,
                quiz,
                st.session_state.tutor_messages,
            )

        st.session_state.tutor_messages.append(
            {"role": "assistant", "content": answer}
        )
        st.toast("답변이 준비되었습니다.", icon="✅")

        if session_id:
            try:
                update_tutor_chat_history(
                    session_id,
                    st.session_state.tutor_messages,
                )
            except Exception:
                st.warning("답변을 Supabase 대화 기록에 저장하지 못했습니다.")

        with st.chat_message("assistant"):
            st.write(answer)
    except Exception as error:
        render_openai_error(error, "답변")


# 채팅 기록과 틀린 문제 복습 및 자유 질문 입력창을 표시한다.
def render_question_tab(summary, quiz):
    if "tutor_messages" not in st.session_state:
        st.session_state.tutor_messages = []

    for message in st.session_state.tutor_messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    wrong_questions = get_wrong_questions(quiz)
    if wrong_questions and st.button("틀린 문제 복습하기"):
        review_prompt = (
            "내가 틀린 문제를 문항 순서대로 하나씩 짚어서, "
            "왜 틀렸는지와 핵심 개념을 쉽게 설명해 주세요."
        )
        submit_tutor_message(
            review_prompt,
            summary,
            quiz,
            "틀린 문제를 복습하고 있습니다...",
        )

    user_prompt = st.chat_input("문서나 퀴즈에 대해 질문해 보세요.")
    if user_prompt:
        submit_tutor_message(
            user_prompt,
            summary,
            quiz,
            "답변을 생성하고 있습니다...",
        )


# 새 PDF를 위한 요약, 퀴즈, 답안, 채팅 상태를 모두 초기화한다.
def reset_tutor_state():
    for key in (
        "summary_file",
        "pdf_page_texts",
        "pdf_text",
        "pdf_summary",
        "pdf_quiz",
        "quiz_answers",
        "tutor_messages",
        "tutor_session_id",
        "restored_session_id",
    ):
        st.session_state.pop(key, None)

    for index in range(3):
        st.session_state.pop(f"quiz_choice_{index}", None)

    if "session_id" in st.query_params:
        del st.query_params["session_id"]


# 저장된 상태를 사용해 요약, 퀴즈, 질문하기 탭을 표시한다.
def render_tutor_tabs(pdf_text=None):
    summary_tab, quiz_tab, question_tab = st.tabs(
        ["요약", "퀴즈", "질문하기"]
    )

    with summary_tab:
        if "pdf_summary" not in st.session_state and pdf_text:
            try:
                with st.spinner("PDF 텍스트를 요약하고 있습니다..."):
                    st.session_state.pdf_summary = summarize_pdf_text(pdf_text)
                    st.toast("요약이 완료되었습니다.", icon="✅")
            except Exception as error:
                render_openai_error(error)

        if "pdf_summary" in st.session_state:
            render_pdf_summary(st.session_state.pdf_summary)
        else:
            st.info("PDF 텍스트가 추출되면 요약이 자동으로 생성됩니다.")

    with quiz_tab:
        if (
            "pdf_summary" in st.session_state
            and "pdf_quiz" not in st.session_state
        ):
            try:
                with st.spinner("요약을 바탕으로 퀴즈를 만들고 있습니다..."):
                    st.session_state.pdf_quiz = generate_quiz(
                        st.session_state.pdf_summary
                    )
                    st.session_state.quiz_answers = {}
                    st.toast("퀴즈가 완료되었습니다.", icon="✅")
            except Exception as error:
                render_openai_error(error, "퀴즈")

        if (
            "pdf_summary" in st.session_state
            and "pdf_quiz" in st.session_state
            and "tutor_session_id" not in st.session_state
        ):
            try:
                with st.spinner("튜터 세션을 Supabase에 저장하고 있습니다..."):
                    session_id = create_tutor_session(
                        st.session_state.pdf_summary,
                        st.session_state.pdf_quiz,
                    )
                    st.session_state.tutor_session_id = session_id
                    st.session_state.tutor_messages = []
                    st.query_params["session_id"] = str(session_id)
            except Exception:
                st.error("튜터 세션을 Supabase에 저장하지 못했습니다.")

        if "pdf_quiz" in st.session_state:
            render_quiz(st.session_state.pdf_quiz)
        elif "pdf_summary" not in st.session_state:
            st.info("요약이 생성되면 퀴즈가 자동으로 생성됩니다.")

    with question_tab:
        if (
            "pdf_summary" in st.session_state
            and "pdf_quiz" in st.session_state
        ):
            render_question_tab(
                st.session_state.pdf_summary,
                st.session_state.pdf_quiz,
            )
        else:
            st.info("요약과 퀴즈가 생성되면 질문할 수 있습니다.")


# 요약 JSON에서 지난 세션 목록에 표시할 짧은 미리보기를 만든다.
def get_summary_preview(summary, max_length=100):
    if not isinstance(summary, dict):
        text = str(summary or "요약 없음")
    else:
        core_summary = summary.get("핵심 요약", [])
        if isinstance(core_summary, list):
            text = " ".join(str(sentence) for sentence in core_summary)
        else:
            text = str(core_summary or "요약 없음")

    return f"{text[:max_length]}…" if len(text) > max_length else text


# 현재 튜터 상태를 비우고 지난 세션 목록 화면으로 이동한다.
def go_to_tutor_list():
    reset_tutor_state()
    st.session_state.tutor_view = "list"
    rerun_app()


# 현재 튜터 상태를 비우고 새 PDF 업로드 화면으로 이동한다.
def start_new_pdf():
    reset_tutor_state()
    st.session_state.tutor_view = "upload"
    st.session_state.uploader_version = (
        st.session_state.get("uploader_version", 0) + 1
    )
    rerun_app()


# Supabase에서 매번 최신 세션 목록을 조회해 표시한다.
def render_tutor_session_list():
    st.header("지난 튜터 세션")
    st.caption("저장된 학습 기록을 다시 열거나 새 PDF 학습을 시작하세요.")

    if st.button("새 PDF로 시작하기", type="primary"):
        start_new_pdf()

    try:
        sessions = fetch_tutor_sessions()
    except Exception:
        st.error("지난 튜터 세션 목록을 불러오지 못했습니다.")
        return

    if not sessions:
        st.info("저장된 튜터 세션이 없습니다.")
        return

    for session in sessions:
        session_id = session["id"]
        content_column, button_column = st.columns([8, 2])

        with content_column:
            st.caption(
                f"생성 시각: {format_created_at(session.get('created_at'))}"
            )
            st.write(get_summary_preview(session.get("summary")))

        with button_column:
            if st.button("세션 열기", key=f"open_session_{session_id}"):
                reset_tutor_state()
                st.query_params["session_id"] = str(session_id)
                st.session_state.tutor_view = "detail"
                rerun_app()

        st.divider()


# URL로 복원한 세션의 요약, 퀴즈, 대화 상세 화면을 표시한다.
def render_tutor_session_detail():
    if st.button("목록으로 돌아가기", key="detail_back_to_list"):
        go_to_tutor_list()

    if (
        "pdf_summary" not in st.session_state
        or "pdf_quiz" not in st.session_state
    ):
        st.info("복원할 튜터 세션을 선택해 주세요.")
        return

    session_id = st.session_state.get("tutor_session_id")
    if session_id is not None:
        st.caption(f"튜터 세션 #{session_id}")

    render_tutor_tabs()


# 새 PDF 업로드와 텍스트 추출 화면을 표시한다.
def render_tutor_upload_page():
    st.header("나만의 튜터")
    st.caption("PDF를 올리면 텍스트 추출부터 요약과 퀴즈까지 자동으로 진행됩니다.")

    if st.button("목록으로 돌아가기", key="upload_back_to_list"):
        go_to_tutor_list()

    uploader_version = st.session_state.get("uploader_version", 0)
    uploaded_pdf = st.file_uploader(
        "PDF 파일 업로드",
        key=f"pdf_uploader_{uploader_version}",
    )

    if uploaded_pdf is None:
        if (
            "pdf_summary" in st.session_state
            and "pdf_quiz" in st.session_state
        ):
            render_tutor_tabs()
        return

    if not uploaded_pdf.name.lower().endswith(".pdf"):
        st.warning("PDF 파일만 업로드할 수 있습니다")
        return

    try:
        summary_file = (uploaded_pdf.name, uploaded_pdf.size)

        if st.session_state.get("summary_file") != summary_file:
            reset_tutor_state()
            page_texts = extract_pdf_pages(uploaded_pdf)
            pages_with_text = [item for item in page_texts if item[1]]

            if not pages_with_text:
                st.warning(
                    "이 PDF에서 텍스트를 찾지 못했습니다. "
                    "이미지로 스캔된 PDF는 별도의 OCR이 필요합니다."
                )
                return

            st.session_state.summary_file = summary_file
            st.session_state.pdf_page_texts = page_texts
            st.session_state.pdf_text = build_pdf_memo_text(pages_with_text)

        page_texts = st.session_state.pdf_page_texts
        pdf_text = st.session_state.pdf_text
        render_pdf_pages(page_texts)

        if st.button("PDF 텍스트를 메모로 저장", type="primary"):
            try:
                add_memo(pdf_text)
                st.success("PDF 텍스트를 메모로 저장했습니다.")
                rerun_app()
            except Exception:
                st.error(
                    "메모를 저장하지 못했습니다. Supabase 연결 설정을 확인해 주세요."
                )

        render_tutor_tabs(pdf_text)
    except Exception as error:
        st.error(f"PDF를 읽는 중 오류가 발생했습니다: {error}")


# 나만의 튜터에서 목록, 업로드, 세션 상세 화면을 전환한다.
def render_pdf_page():
    view = st.session_state.get("tutor_view", "list")

    if view == "upload":
        render_tutor_upload_page()
    elif view == "detail":
        render_tutor_session_detail()
    else:
        render_tutor_session_list()


# 앱을 초기화하고 사이드바 선택에 맞는 화면을 표시한다.
def main():
    st.set_page_config(
        page_title="나만의 메모장",
        page_icon="📝",
        layout="wide",
    )
    apply_custom_styles() 
    st.title("📝 나만의 메모장")
    st.caption("메모 기록부터 PDF 학습까지 한곳에서 관리하세요.")

    restore_tutor_session_from_query()
    selected_menu = render_sidebar()

    if selected_menu == "메모":
        render_memo_page()
    else:
        render_pdf_page()


main()
