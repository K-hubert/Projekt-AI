import streamlit as st
from rag.rag_core import RagIndex, answer

st.set_page_config(page_title="PDF RAG Chat", layout="wide")

st.title("PDF RAG Chat")
st.caption("Wrzuć PDF(y) → indeksuję → pytasz → dostajesz odpowiedź ze źródłami.")

# Session state
if "rag" not in st.session_state:
    st.session_state.rag = None
if "ready" not in st.session_state:
    st.session_state.ready = False
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("Ustawienia")
    mode = st.selectbox("Tryb odpowiedzi", ["qa", "study"], format_func=lambda x: "Q&A" if x=="qa" else "Notatka do nauki")
    top_k = st.slider("TOP_K (ile fragmentów brać)", 1, 10, 4)
    llm_model = st.text_input("Model LLM", value="gpt-4o-mini")
    st.divider()
    uploaded = st.file_uploader("Wybierz PDF(y)", type=["pdf"], accept_multiple_files=True)

    col1, col2 = st.columns(2)
    build = col1.button("Zbuduj indeks", use_container_width=True)
    reset = col2.button("Reset", use_container_width=True)

    if reset:
        st.session_state.rag = None
        st.session_state.ready = False
        st.session_state.messages = []
        st.rerun()

    if build:
        if not uploaded:
            st.warning("Wrzuć najpierw przynajmniej jeden PDF.")
        else:
            with st.spinner("Indeksuję PDF(y)..."):
                files = [(f.name, f.getvalue()) for f in uploaded]
                rag = RagIndex()
                rag.build_from_uploaded_pdfs(files)
                st.session_state.rag = rag
                st.session_state.ready = True
            st.success("Gotowe. Możesz zadawać pytania.")

# Chat
if not st.session_state.ready:
    st.info("Wrzuć PDF(y) w sidebarze i kliknij **Zbuduj indeks**.")
else:
    # show history
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    user_q = st.chat_input("Zadaj pytanie do PDF...")
    if user_q:
        st.session_state.messages.append({"role": "user", "content": user_q})
        with st.chat_message("user"):
            st.markdown(user_q)

        with st.chat_message("assistant"):
            with st.spinner("Szukam w PDF i składam odpowiedź..."):
                resp = answer(
                    question=user_q,
                    rag=st.session_state.rag,
                    mode=mode,
                    top_k=top_k,
                    llm_model=llm_model,
                )
                st.markdown(resp)

        st.session_state.messages.append({"role": "assistant", "content": resp})
