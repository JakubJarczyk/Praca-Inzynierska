import os
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Katalog z plikami danych
PICKLE_DIR = 'datasets_pickle'

# Wyświetlanie dostępnych plików w katalogu
print("Dostępne pliki w katalogu './datasets_pickle':")
pickle_files = os.listdir(PICKLE_DIR)
train_files = [f for f in pickle_files if 'train' in f]
test_files = [f for f in pickle_files if 'test' in f]

if not train_files or not test_files:
    print("Brak odpowiednich plików Pickle dla danych treningowych lub testowych.")
    exit()

print("\nPliki treningowe:")
for idx, file in enumerate(train_files, start=1):
    print(f"{idx}. {file}")

try:
    selected_train_index = int(input("Wybierz numer pliku treningowego: ")) - 1
    if selected_train_index < 0 or selected_train_index >= len(train_files):
        raise ValueError("Wybrano nieprawidłowy numer pliku treningowego.")
    selected_train_file = train_files[selected_train_index]
except ValueError as e:
    print(f"Błąd: {e}. Kończę program.")
    exit()

print("\nPliki testowe:")
for idx, file in enumerate(test_files, start=1):
    print(f"{idx}. {file}")

try:
    selected_test_index = int(input("Wybierz numer pliku testowego: ")) - 1
    if selected_test_index < 0 or selected_test_index >= len(test_files):
        raise ValueError("Wybrano nieprawidłowy numer pliku testowego.")
    selected_test_file = test_files[selected_test_index]
except ValueError as e:
    print(f"Błąd: {e}. Kończę program.")
    exit()

train_path = os.path.join(PICKLE_DIR, selected_train_file)
test_path = os.path.join(PICKLE_DIR, selected_test_file)

print(f"Wczytuję dane treningowe z pliku: {train_path}")
print(f"Wczytuję dane testowe z pliku: {test_path}")

# Wczytanie danych
with open(train_path, 'rb') as f:
    train_data_dict = pickle.load(f)
with open(test_path, 'rb') as f:
    test_data_dict = pickle.load(f)

x_train = np.asarray(train_data_dict['data'])
y_train = np.asarray(train_data_dict['labels'])
x_test = np.asarray(test_data_dict['data'])
y_test = np.asarray(test_data_dict['labels'])

# Wykrywanie liczby cech na punkt dłoni (21 punktów w MediaPipe)
num_features_per_landmark = len(x_train[0]) // 21
print(f"\nLiczba cech na punkt dłoni: {num_features_per_landmark}")

# Jeśli dane mają wymiar 'z', przygotuj zestaw bez wymiaru 'z'
test_variants = {}
if num_features_per_landmark == 3:
    print("Dane zawierają współrzędne 'x', 'y' oraz 'z'.")
    x_train_without_z = np.delete(x_train, np.arange(2, x_train.shape[1], 3), axis=1)
    x_test_without_z = np.delete(x_test, np.arange(2, x_test.shape[1], 3), axis=1)
    test_variants = {
        "Dane z wymiarem 'z'": (x_train, x_test),
        "Dane bez wymiaru 'z'": (x_train_without_z, x_test_without_z)
    }
elif num_features_per_landmark == 2:
    print("Dane zawierają tylko współrzędne 'x' i 'y'.")
    test_variants = {
        "Dane bez wymiaru 'z'": (x_train, x_test)
    }
else:
    print("Nieprawidłowy format danych. Kończę program.")
    exit()

# Lista klasyfikatorów do przetestowania
classifiers = {
    "Random Forest": RandomForestClassifier(),
    "Support Vector Machine": SVC(),
    "Decision Tree": DecisionTreeClassifier(),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Logistic Regression": LogisticRegression(max_iter=1000)
}

# Funkcja do rysowania macierzy pomyłek
def plot_confusion_matrix(y_true, y_pred, classes, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('Prawdziwe etykiety')
    plt.xlabel('Przewidywane etykiety')
    plt.tight_layout()
    plt.show()

# Testowanie klasyfikatorów dla każdej wersji danych
for variant_name, (variant_x_train, variant_x_test) in test_variants.items():
    print(f"\nTestowanie klasyfikatorów dla: {variant_name}")

    results = {}
    for name, clf in classifiers.items():
        # Trening klasyfikatora
        clf.fit(variant_x_train, y_train)

        # Predykcja na danych testowych
        y_predict = clf.predict(variant_x_test)

        # Obliczanie metryk
        accuracy = accuracy_score(y_test, y_predict)
        print(f"\n{name}:")
        print(f"Accuracy: {accuracy * 100:.2f}%")
        print("Classification Report:")
        print(classification_report(y_test, y_predict))

        # Rysowanie macierzy pomyłek
        plot_confusion_matrix(y_test, y_predict, classes=np.unique(y_test), title=f"Macierz pomyłek - {name} ({variant_name})")

        # Zapisywanie wyników
        results[name] = {
            "accuracy": accuracy,
            "classification_report": classification_report(y_test, y_predict, output_dict=True)
        }

    # Wyświetlanie najlepszego klasyfikatora dla tej wersji danych
    best_model = max(results, key=lambda x: results[x]['accuracy'])
    print(f"\nNajlepszy klasyfikator dla {variant_name}:")
    print(f"{best_model} z dokładnością {results[best_model]['accuracy'] * 100:.2f}%")
