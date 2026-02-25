import os
import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Katalog z plikami pickle
PICKLE_DIR = './datasets_pickle'

# Lista dostępnych plików
print("Dostępne pliki w katalogu './datasets_pickle':")
files = [f for f in os.listdir(PICKLE_DIR) if f.endswith('.pickle')]
for idx, file in enumerate(files, start=1):
    print(f"{idx}. {file}")

# Wybór pliku do analizy
try:
    selected_index = int(input("Wybierz numer pliku do analizy: ")) - 1
    if selected_index < 0 or selected_index >= len(files):
        raise ValueError("Wybrano nieprawidłowy numer pliku.")
    selected_file = files[selected_index]
    selected_file_path = os.path.join(PICKLE_DIR, selected_file)
except ValueError as e:
    print(f"Błąd: {e}. Kończę program.")
    exit()

# Wczytanie danych z pliku
print(f"\nWczytuję dane z pliku: {selected_file_path}")
with open(selected_file_path, 'rb') as f:
    data_dict = pickle.load(f)

# Sprawdzenie kluczy w danych
if not isinstance(data_dict, dict) or 'data' not in data_dict or 'labels' not in data_dict:
    print("Błąd: Oczekiwano danych w formacie słownika z kluczami 'data' i 'labels'.")
    exit()

# Tworzenie DataFrame z danych
df = pd.DataFrame(data_dict['data'])
df['label'] = data_dict['labels']

# Podstawowe informacje o danych
print("\nPodstawowe informacje o danych:")
print(df.info())

# Podstawowe statystyki opisowe
print("\nPodstawowe statystyki opisowe:")
print(df.describe())

# Rozkład etykiet
print("\nRozkład etykiet w zbiorze danych:")
label_distribution = df['label'].value_counts()
print(label_distribution)

plt.figure(figsize=(10, 6))
label_distribution.plot(kind='bar')
plt.title('Rozkład etykiet (rzeczywiste klasy)')
plt.xlabel('Etykiety klas')
plt.ylabel('Liczba próbek')
plt.show()

# Obliczenie i wizualizacja macierzy korelacji
# Wybieramy tylko kolumny numeryczne
numeric_columns = df.select_dtypes(include=['float64']).columns
correlation_matrix = df[numeric_columns].corr()

print("\nMacierz korelacji obliczona dla danych numerycznych:")
print(correlation_matrix)

plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', fmt='.2f')
plt.title('Macierz korelacji dla danych numerycznych')
plt.show()
