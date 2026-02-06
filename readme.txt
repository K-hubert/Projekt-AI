PDF RAG Chat – Chatbot oparty o Retrieval Augmented Generation

Opis projektu

Projekt PDF RAG Chat jest aplikacją typu chatbot wykorzystującą technikę Retrieval Augmented Generation (RAG) do generowania odpowiedzi na pytania użytkownika w oparciu o dostarczone dokumenty PDF.
System łączy semantyczne wyszukiwanie w bazie wektorowej z generowaniem odpowiedzi przez duży model językowy (LLM), zapewniając odpowiedzi ograniczone wyłącznie do treści dokumentów źródłowych oraz opatrzone wskazaniem źródeł.

Projekt został zrealizowany w ramach zaliczenia przedmiotu z zakresu Generative AI / LLM / NLP.

Zakres funkcjonalny

Aplikacja umożliwia:
wczytywanie dokumentów PDF
automatyczne dzielenie tekstu na chunki i wektoryzację
semantyczne wyszukiwanie fragmentów dokumentów (Similarity / MMR)
generowanie odpowiedzi z użyciem LLM na podstawie odnalezionego kontekstu
prezentację źródeł (plik, strona, chunk)
generowanie fiszek edukacyjnych
analizę statystyczną danych wejściowych
ewaluację jakości odpowiedzi na podstawie zdefiniowanego zbioru testowego

Architektura systemu

Projekt został zaprojektowany zgodnie z zasadami programowania obiektowego (OOP) i podzielony na czytelne komponenty.

Główne moduły

RagIndex
Odpowiada za przetwarzanie dokumentów PDF, dzielenie tekstu na fragmenty, generowanie embeddingów, przechowywanie metadanych oraz budowę i obsługę indeksu wektorowego FAISS.

Retriever
Odpowiada za semantyczne wyszukiwanie kontekstu w bazie wektorowej z wykorzystaniem wyszukiwania similarity oraz metody MMR (Maximal Marginal Relevance).

PromptBuilder
Buduje prompt systemowy w zależności od wybranego trybu odpowiedzi (QA lub notatka do nauki), stylu odpowiedzi, szablonu promptu oraz użycia techniki few-shot learning.

Evaluator
Minimalny moduł ewaluacyjny sprawdzający obecność sekcji „Źródła”, wykorzystanie kontekstu oraz poprawność strukturalną odpowiedzi.

Interfejs użytkownika

Aplikacja posiada interfejs webowy zbudowany w Streamlit, podzielony na trzy zakładki.

1. Chat
Umożliwia zadawanie pytań do dokumentów PDF, wybór trybu odpowiedzi (QA lub notatka do nauki), wybór stylu odpowiedzi (krótka odpowiedź, notatka do nauki, odpowiedź egzaminacyjna), wybór szablonu promptu, włączenie lub wyłączenie few-shot learning oraz podgląd historii rozmowy.

2. Fiszki
Umożliwia generowanie fiszek edukacyjnych na podstawie dokumentów oraz eksport fiszek do pliku CSV.

3. Analiza
Prezentuje statystyki danych wejściowych, takie jak liczba plików, fragmentów tekstu i słów, najczęściej występujące terminy oraz histogram długości chunków.

Ewaluacja i testy

Projekt zawiera moduł automatycznej ewaluacji odpowiedzi generowanych przez system.


Zbiór testowy

Testy definiowane są w pliku:
tests/eval_questions.json

Każde pytanie zawiera treść pytania oraz listę oczekiwanych słów kluczowych wykorzystywanych do oceny poprawności odpowiedzi.

Uruchomienie testów
python -m tests.run_eval

Wyniki ewaluacji

Wyniki testów są wyświetlane w konsoli oraz zapisywane do pliku:
tests/eval_report.csv

Ewaluacja obejmuje obecność źródeł w odpowiedzi, wykorzystanie kontekstu, pokrycie słów kluczowych oraz porównanie metod wyszukiwania dokumentów (similarity vs MMR).

Technologie

Python 3.11
Streamlit
OpenAI API
FAISS (indeksowanie i wyszukiwanie wektorowe)
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
│   └── run_eval.py
│
├── requirements.txt
└── README.txt