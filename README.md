# MNIST Digit Classification using CNN (PyTorch)

Projekt przedstawia kompletny proces klasyfikacji ręcznie pisanych cyfr ze zbioru MNIST, wykorzystując konwolucyjną sieć neuronową (CNN) zaimplementowaną w bibliotece PyTorch.

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
 
## Architektura modelu

- Warstwa konwolucyjna `Conv2d` z 32 filtrami, jądrem 3×3  
- Warstwa MaxPooling 2×2  
- Warstwa w pełni połączona (FC) o 128 neuronach  
- Wyjście: warstwa FC o 10 neuronach (klasy cyfr 0-9)  
- Funkcja aktywacji: ReLU  
- Na wyjściu: logarytmiczna funkcja softmax (LogSoftmax)  
- Optymalizator: Adam  
- Funkcja straty: NLLLoss (Negative Log Likelihood)  

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
