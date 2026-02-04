import streamlit as st
import pandas as pd

from rag.rag_core import RagIndex, answer, generate_flashcards


# Page config

st.set_page_config(
    page_title="PDF RAG Chat",
    layout="wide",
)

st.title("PDF RAG Chat")
st.caption("Wrzuć PDF(y) → indeksuję → pytasz → dostajesz odpowiedź + fiszki do nauki.")


# Session state init

if "rag" not in st.session_state:
    st.session_state.rag = None

if "ready" not in st.session_state:
    st.session_state.ready = False

if "messages" not in st.session_state:
    st.session_state.messages = []

if "flashcards" not in st.session_state:
    st.session_state.flashcards = None


# Sidebar

with st.sidebar:
    st.header("Ustawienia")

    mode = st.selectbox(
        "Tryb odpowiedzi",
        ["qa", "study"],
        format_func=lambda x: "Q&A" if x == "qa" else "Notatka do nauki",
    )

    top_k = st.slider("TOP_K (ile fragmentów brać)", 1, 10, 4)

    llm_model = st.text_input("Model LLM", value="gpt-4o-mini")

    st.divider()

    uploaded_files = st.file_uploader(
        "Wybierz PDF(y)",
        type=["pdf"],
        accept_multiple_files=True,
    )

    col1, col2 = st.columns(2)
    build_index = col1.button("Zbuduj indeks", use_container_width=True)
    reset_all = col2.button("Reset", use_container_width=True)

    st.divider()
    st.subheader("Fiszki do nauki")

    n_cards = st.slider("Liczba fiszek", 5, 60, 20)
    gen_cards = st.button("Generuj fiszki", use_container_width=True)


# Reset logic

if reset_all:
    st.session_state.rag = None
    st.session_state.ready = False
    st.session_state.messages = []
    st.session_state.flashcards = None
    st.rerun()

# Build index

if build_index:
    if not uploaded_files:
        st.warning("Wrzuć przynajmniej jeden PDF.")
    else:
        with st.spinner("Indeksuję PDF(y)..."):
            files = [(f.name, f.getvalue()) for f in uploaded_files]
            rag = RagIndex()
            rag.build_from_uploaded_pdfs(files)
            st.session_state.rag = rag
            st.session_state.ready = True
            st.session_state.messages = []
            st.session_state.flashcards = None
        st.success("Indeks gotowy. Możesz zadawać pytania.")


# Generate flashcards

if gen_cards:
    if not st.session_state.ready:
        st.warning("Najpierw zbuduj indeks.")
    else:
        with st.spinner("Generuję fiszki do nauki..."):
            cards = generate_flashcards(
                rag=st.session_state.rag,
                n_cards=n_cards,
                llm_model=llm_model,
            )
            st.session_state.flashcards = cards
        st.success("Fiszki wygenerowane.")

# Flashcards

if st.session_state.flashcards:
    st.subheader("Fiszki egzaminacyjne")

    cards = st.session_state.flashcards

    for i, c in enumerate(cards, 1):
        with st.expander(f"{i}. {c.get('question', '')}"):
            st.markdown(c.get("answer", ""))
            if c.get("sources"):
                st.caption("Źródła: " + " | ".join(c["sources"]))

    df = pd.DataFrame(
        [
            {
                "question": c.get("question", ""),
                "answer": c.get("answer", ""),
                "sources": " | ".join(c.get("sources", [])),
            }
            for c in cards
        ]
    )

    st.download_button(
        "Pobierz fiszki jako CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="flashcards.csv",
        mime="text/csv",
        use_container_width=True,
    )

    st.divider()

# --------------------------------------------------
# Chat
# --------------------------------------------------
if not st.session_state.ready:
    st.info("Wrzuć PDF(y) w sidebarze i kliknij **Zbuduj indeks**.")
else:
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
