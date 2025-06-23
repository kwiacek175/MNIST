# MNIST Digit Classification using CNN (PyTorch)

Projekt ten przedstawia kompletny pipeline klasyfikacji cyfr ręcznie pisanych (MNIST) z wykorzystaniem konwolucyjnej sieci neuronowej (CNN) zaimplementowanej w PyTorch.

## Cel projektu

- Zapoznanie się z podstawami uczenia maszynowego na klasycznym zbiorze danych MNIST
- Implementacja prostej architektury CNN do rozpoznawania cyfr 0-9
- Wizualizacja procesu uczenia (strata, dokładność, zmiany wag)
- Analiza wyników poprzez macierz konfuzji i przykładowe predykcje

---

## Zawartość

- ✅ Przygotowanie danych z `torchvision.datasets.MNIST`
- ✅ Architektura CNN z 1 warstwą konwolucyjną + 2 w pełni połączonymi
- ✅ Trenowanie i ewaluacja modelu (accuracy, loss)
- ✅ Wizualizacja:
  - funkcji straty,
  - skuteczności klasyfikacji (accuracy),
  - zmian wag w czasie,
  - macierzy konfuzji,
  - rozkładu prawdopodobieństw dla przykładowych obrazów

## Wymagania

- Python 3.8+
- PyTorch
- torchvision
- matplotlib
- numpy
- scikit-learn

Zainstaluj wszystko komendą:

```bash
pip install torch torchvision matplotlib numpy scikit-learn
