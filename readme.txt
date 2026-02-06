PDF RAG Chat – Chatbot oparty o Retrieval Augmented Generation
Opis projektu

Projekt PDF RAG Chat jest aplikacją typu chatbot, wykorzystującą technikę Retrieval Augmented Generation (RAG) do generowania odpowiedzi na pytania użytkownika w oparciu o dostarczone dokumenty PDF.
System łączy wyszukiwanie semantyczne w bazie wektorowej z generowaniem odpowiedzi przez duży model językowy (LLM), zapewniając odpowiedzi ograniczone wyłącznie do treści dokumentów źródłowych oraz opatrzone wskazaniem źródeł.

Projekt został zrealizowany w ramach zaliczenia przedmiotu z zakresu Generative AI / LLM / NLP.


Zakres funkcjonalny

Aplikacja umożliwia:

wczytywanie dokumentów PDF,

automatyczne dzielenie tekstu na chunki i wektoryzację,

semantyczne wyszukiwanie fragmentów dokumentów (Similarity / MMR),

generowanie odpowiedzi z użyciem LLM na podstawie odnalezionego kontekstu,

prezentację źródeł (plik, strona, chunk),

generowanie fiszek edukacyjnych,

analizę statystyczną danych wejściowych,

ewaluację jakości odpowiedzi na podstawie zdefiniowanego zbioru testowego.


Architektura systemu

Projekt został zaprojektowany zgodnie z zasadami programowania obiektowego (OOP) i podzielony na czytelne komponenty:

Główne moduły

RagIndex
Odpowiada za:

przetwarzanie PDF,

chunkowanie tekstu,

wektoryzację,

przechowywanie metadanych,

budowę i obsługę indeksu semantycznego.

Retriever
Odpowiada za wyszukiwanie kontekstu:

similarity search,

MMR (Maximal Marginal Relevance).

PromptBuilder
Buduje prompt systemowy w zależności od:

trybu (QA / notatka / egzamin),

stylu odpowiedzi,

szablonu promptu,

użycia few-shot learning.

Evaluator
Minimalny moduł ewaluacyjny sprawdzający:

obecność sekcji „Źródła”,

wykorzystanie kontekstu,

poprawność strukturalną odpowiedzi.

Interfejs użytkownika

Aplikacja posiada interfejs webowy zbudowany w Streamlit, podzielony na trzy zakładki:

1. Chat

zadawanie pytań do PDF,

wybór stylu odpowiedzi:

Krótka odpowiedź,

Notatka do nauki,

Odpowiedź egzaminacyjna,

wybór szablonu promptu,

włączenie/wyłączenie few-shot learning,

podgląd historii rozmowy.

2. Fiszki

generowanie fiszek edukacyjnych na podstawie dokumentów,

eksport fiszek do pliku CSV.

3. Analiza

statystyki danych wejściowych (liczba plików, chunków, słów),

najczęstsze terminy,

histogram długości chunków.

Ewaluacja i testy

Projekt zawiera moduł automatycznej ewaluacji odpowiedzi.

Zbiór testowy

Testy definiowane są w pliku:

tests/eval_questions.json


Każde pytanie zawiera:

treść pytania,

listę oczekiwanych słów kluczowych,

informację o konieczności cytowania źródeł.

Uruchomienie testów
python -m tests.run_eval

Wyniki ewaluacji

wyświetlane w konsoli,

zapisywane do pliku:

tests/eval_report.csv


Ewaluacja obejmuje:

obecność źródeł,

wykorzystanie kontekstu,

pokrycie słów kluczowych,

porównanie metod wyszukiwania (similarity vs MMR).

Technologie

Python 3.11

Streamlit

OpenAI API

FAISS (wektoryzacja)

NumPy

PyMuPDF

Matplotlib

JSON / CSV

Struktura projektu
Projekt-AI/
│
├── app.py                 # aplikacja Streamlit
├── rag/
│   ├── rag_core.py        # logika RAG
│   ├── retriever.py       # wyszukiwanie kontekstu
│   ├── prompt_builder.py # budowa promptów
│   └── evaluator.py      # ewaluacja odpowiedzi
│
├── data/
│   └── pdfs/              # PDF-y do testów
│
├── tests/
│   ├── eval_questions.json
│   ├── run_eval.py
    └── data.py
│
├── requirements.txt
└── README.txt

Status projektu

✔ działający system RAG
✔ OOP
✔ analiza danych
✔ ewaluacja i testy
✔ system kontroli wersji (GitHub)