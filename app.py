import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from rag.rag_core import RagIndex, answer, generate_flashcards, compute_text_stats
from rag.evaluator import Evaluator

st.set_page_config(page_title="PDF RAG Chat", layout="wide")

st.title("PDF RAG Chat")
st.caption("Wrzuć PDF(y) → indeksuję → pytasz → odpowiedź ze źródłami + fiszki + analiza.")


# Session state

if "rag" not in st.session_state:
    st.session_state.rag = None
if "ready" not in st.session_state:
    st.session_state.ready = False
if "messages" not in st.session_state:
    st.session_state.messages = []
if "flashcards" not in st.session_state:
    st.session_state.flashcards = None
if "stats" not in st.session_state:
    st.session_state.stats = None


# Sidebar

with st.sidebar:
    st.header("Ustawienia")

    mode = st.selectbox(
        "Tryb odpowiedzi",
        ["qa", "study"],
        format_func=lambda x: "Q&A" if x == "qa" else "Notatka do nauki",
    )

    top_k = st.slider("TOP_K (ile fragmentów brać)", 1, 10, 4)

    retrieval_method = st.selectbox(
        "Dopasowanie dokumentów",
        ["similarity", "mmr"],
        format_func=lambda x: "Similarity (najbliższe)" if x == "similarity" else "MMR (różnorodne)",
    )

    mmr_lambda = st.slider("MMR λ (balans trafność/różnorodność)", 0.0, 1.0, 0.6, 0.05)
    fetch_k = st.slider("MMR fetch_k (kandydaci)", 10, 80, 30, 5)

    llm_model = st.text_input("Model LLM", value="gpt-4o-mini")

    st.divider()

    uploaded_files = st.file_uploader(
        "Wybierz PDF(y)",
        type=["pdf"],
        accept_multiple_files=True,
    )

    col1, col2 = st.columns(2)
    build_index = col1.button("Zbuduj indeks", width='stretch')
    reset_all = col2.button("Reset", width='stretch')

    st.divider()
    st.subheader("Fiszki do nauki")
    n_cards = st.slider("Liczba fiszek", 5, 60, 20)
    gen_cards = st.button("Generuj fiszki", width='stretch')


# Reset

if reset_all:
    st.session_state.rag = None
    st.session_state.ready = False
    st.session_state.messages = []
    st.session_state.flashcards = None
    st.session_state.stats = None
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

            # statystyki liczymy od razu po indeksacji
            st.session_state.stats = compute_text_stats(rag.meta)

        st.success("Indeks gotowy.")


# Generate flashcards

if gen_cards:
    if not st.session_state.ready:
        st.warning("Najpierw zbuduj indeks.")
    else:
        with st.spinner("Generuję fiszki..."):
            cards = generate_flashcards(
                rag=st.session_state.rag,
                n_cards=n_cards,
                llm_model=llm_model,
            )
            st.session_state.flashcards = cards
        st.success("Fiszki gotowe.")

# Tabs

tab_chat, tab_flashcards, tab_analysis = st.tabs(["Chat", "Fiszki", "Analiza"])


# TAB Chat

# TAB Chat

with tab_chat:
    if not st.session_state.ready:
        st.info("Wrzuć PDF(y) w sidebarze i kliknij **Zbuduj indeks**.")
    else:
        # 1) INPUT NA GÓRZE (zamiast st.chat_input)
        with st.form("chat_form", clear_on_submit=True):
            profile_ui = st.selectbox(
                "Styl odpowiedzi",
                ["Krótka odpowiedź", "Notatka do nauki", "Odpowiedź egzaminacyjna"],
                index=0
            )

            profile_map = {
                "Krótka odpowiedź": "concise",
                "Notatka do nauki": "study",
                "Odpowiedź egzaminacyjna": "exam"
            }
            profile = profile_map[profile_ui]

            prompt_template = st.selectbox(
                "Szablon promptu",
                ["Standard", "Ścisły (tylko z kontekstu)", "Nauczyciel (wyjaśnij prosto)"],
                index=0
            )
            use_few_shot_ui = st.checkbox("Few-shot (przykłady w promptcie)", value=True)
            user_q = st.text_input("Zadaj pytanie do PDF...", placeholder="Np. Z czego składa się komputer?")
            submitted = st.form_submit_button("Wyślij")

        # 2) Obsługa wysłania (generujemy odpowiedź i dopisujemy do historii)
        if submitted and user_q.strip():
            st.session_state.messages.append({"role": "user", "content": user_q})

            with st.spinner("Szukam w PDF i składam odpowiedź..."):
                resp = answer(
                    question=user_q,
                    rag=st.session_state.rag,
                    mode=mode,
                    top_k=top_k,
                    llm_model=llm_model,
                    retrieval_method=retrieval_method,
                    mmr_lambda=mmr_lambda,
                    fetch_k=fetch_k,
                    profile=profile,
                    prompt_template=prompt_template,
                    use_few_shot=use_few_shot_ui,
                )
            evaluator = Evaluator()
            retrieved = getattr(st.session_state.rag, "last_retrieved", [])
            eval_res = evaluator.evaluate(resp, retrieved)

            
            label = f"⚙️ Styl: {profile_ui} | 🧩 Szablon: {prompt_template} | 🎓 Few-shot: {'TAK' if use_few_shot_ui else 'NIE'} | 🔎 Retrieval: {retrieval_method}"
            st.session_state.messages.append(
                {"role": "assistant", "content": f"{label}\n\n{resp}"}
            )


            st.rerun()

        st.divider()

        # 3) HISTORIA POD SPodem, w kolejności (najstarsze u góry, nowe niżej)
        for m in reversed(st.session_state.messages):
            with st.chat_message(m["role"]):
                st.markdown(m["content"])

# TAB Flashcards

with tab_flashcards:
    if not st.session_state.ready:
        st.info("Najpierw zbuduj indeks PDF.")
    else:
        st.write("Generuj fiszki przyciskiem w sidebarze.")
        if st.session_state.flashcards:
            cards = st.session_state.flashcards
            for i, c in enumerate(cards, 1):
                with st.expander(f"{i}. {c.get('question', '')}"):
                    st.markdown(c.get("answer", ""))
                    if c.get("sources"):
                        st.caption("Źródła: " + " | ".join(c["sources"]))

            df = pd.DataFrame(
                [{
                    "question": c.get("question", ""),
                    "answer": c.get("answer", ""),
                    "sources": " | ".join(c.get("sources", []))
                } for c in cards]
            )

            st.download_button(
                "Pobierz fiszki jako CSV",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name="flashcards.csv",
                mime="text/csv",
                width="stretch",

            )
        else:
            st.warning("Brak fiszek. Kliknij **Generuj fiszki** w sidebarze.")


# TAB Analysis

with tab_analysis:
    if not st.session_state.ready:
        st.info("Najpierw zbuduj indeks PDF.")
    else:
        stats = st.session_state.stats or {}
        st.subheader("Statystyki tekstu")

        colA, colB, colC, colD = st.columns(4)
        colA.metric("Pliki", stats.get("n_sources", 0))
        colB.metric("Chunky", stats.get("n_chunks", 0))
        colC.metric("Znaki", stats.get("total_chars", 0))
        colD.metric("Słowa", stats.get("total_words", 0))

        colE, colF, colG = st.columns(3)
        colE.metric("Śr. słów/chunk", stats.get("avg_words_per_chunk", 0))
        colF.metric("Min słów/chunk", stats.get("min_words_per_chunk", 0))
        colG.metric("Max słów/chunk", stats.get("max_words_per_chunk", 0))

        st.divider()
        st.subheader("Najczęstsze terminy")
        top_terms = stats.get("top_terms", [])
        if top_terms:
            st.dataframe(pd.DataFrame(top_terms, columns=["term", "count"]), width='stretch')
        else:
            st.write("Brak danych do pokazania.")

        st.divider()

        # Rozkład długości chunków (histogram)
        st.subheader("Rozkład długości chunków (histogram)")

        chunk_lengths = [
            len((c.get("text", "") if isinstance(c, dict) else c.text).split())
            for c in st.session_state.rag.meta
        ]

        fig, ax = plt.subplots()
        ax.hist(chunk_lengths, bins=30)
        ax.set_xlabel("Liczba słów w chunku")
        ax.set_ylabel("Liczba chunków")
        st.pyplot(fig)
