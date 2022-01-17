INSTRUKCJA

1. Rozpakować archiwum
2. W głównym katalogu utworzyć virtualne śrdowowiko Python `virtualenv venv`
3. Aktywować wirtualne środowisko `venv\Scripts\activate`
4. Zainstalować wymagane biblioteki `python -m pip install -r requirements.txt`
5. Wejść do katalogu "app" i uruchomić plik "startup.py" `python startup.py`

UWAGA

Program wyrzuca błąd na koniec swojego działania (assertion error).
Jest to normalne działanie programu i nie należy się traktować tego jako faktyczny błąd.

Spis temperatur w każdej z iteracji znajduje się w katalogu "app" (inaczej niż
w sprawozdaniu)