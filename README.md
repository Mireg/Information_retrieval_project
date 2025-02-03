# O Systemie Analizy Muzyki

## Przegląd Projektu
System analizy recenzji muzycznych zbudowany w oparciu o Flask, przetwarzający ponad 20 000 recenzji z Rate Your Music. Recenzje nie zawierają informacji o albumie, który został oceniony, więc system przypisuje przewidywane albumy na podstawie kontekstu. Zestaw danych zawiera drugą tabelę SQL z danymi dotyczącymi 5000 najczęściej ocenianych albumów na RYM.

---

## Szczegóły Implementacji & Analiza Punktów

| Kategoria                  | Podkategoria               | Dostępne Punkty | Zdobyte Punkty | Uzasadnienie                                                                 |
|----------------------------|----------------------------|-----------------|----------------|------------------------------------------------------------------------------|
| Aplikacja Webowa           | Implementacja Flask        | 15              | 15             | Pełna implementacja frameworku Flask                                        |
| Rozmiar Zestawu Danych     | Liczba Rekordów            | 10              | 10             | ~25 000 rekordów łącznie                                                    |
| Przechowywanie Danych      | Baza Danych SQL            | 10              | 10             | SQLite z SQLAlchemy                                                         |
| Unikalne Funkcje           | Liczba Pól                 | 10              | 8              | 16 unikalnych pól w tabelach Albumów i Recenzji                             |
| Różnorodność Źródeł Danych | Więcej niż Jedno Źródło    | 15              | 5              | Dwa zestawy danych (albums.csv, reviews.csv)                                |
| Poprawa Jakości Danych     | Ulepszenia Pól             | 10              | 6              | Analiza sentymentu, wykrywanie języka, ocena pewności                       |
| Filtry Wyszukiwania        | Opcje Filtrowania          | 10              | 6              | Język, rok, ocena, gatunek, metoda podobieństwa                             |
| Generowanie Tekstu         | Analiza Oparta na Regułach | 10              | 5              | Podstawowa implementacja analizy sentymentu                                 |
| Waga Terminów              | TF-IDF                     | 15              | 10             | Wektoryzacja TF-IDF dla wyszukiwania                                        |
| Podobieństwo Terminów      | Miary Podobieństwa         | 12              | 5              | Podstawowe dopasowanie rozmyte (fuzzywuzzy)                                 |
| Trafność Wyszukiwania      | Miary Efektywności         | 10              | 5              | Podstawowe ocenianie podobieństwa                                           |
| Podobieństwo Dokumentów    | Metryki Podobieństwa       | 20              | 15             | Podobieństwo Cosinusowe, Dice, Jaccard                                      |
| Analiza Tekstu             | Funkcje Analizy            | 40              | 20             | Analiza sentymentu, wykrywanie języka                                       |
| Wizualizacja               | Typy Wykresów              | 20              | 12             | Interaktywne rozkłady ocen, gatunków, lat                                   |
| Implementacja Cache        | Przechowywanie Wyników     | 5               | 5              | Implementacja Flask-Caching                                                 |

---

### **Łączna Liczba Punktów: 136**