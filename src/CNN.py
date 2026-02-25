import os
import cv2
import numpy as np
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Katalog główny z danymi
CROPPED_HANDS_DIR = 'skadrowane_dlonie'

# Pytanie o liczbę danych do przetworzenia
use_all = input("Czy chcesz przetworzyć wszystkie dane? (tak/nie): ").lower() == 'tak'
data_limit = None if use_all else int(input("Podaj maksymalną liczbę zdjęć do przetworzenia: "))

# Definicja macierzy jądra (kernel) do wyostrzania obrazów
sharpen_kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]], dtype=np.float32)

# Funkcja do wczytywania i przetwarzania zdjęć
def load_data(data_dir, subset_type, data_limit=None):
    data = []
    labels = []
    class_names = []  # Lista nazw klas

    for class_name in os.listdir(data_dir):
        class_dir = os.path.join(data_dir, class_name)
        subset_dir = os.path.join(class_dir, subset_type)  # train/test

        if not os.path.isdir(subset_dir):
            print(f"Pomijam '{subset_dir}', nie znaleziono katalogu '{subset_type}'.")
            continue

        print(f"Wczytuję zdjęcia klasy: {class_name} ({subset_type})")
        class_names.append(class_name)

        images = os.listdir(subset_dir)
        random.shuffle(images)  # Losowe przetasowanie obrazów
        if data_limit:
            images = images[:data_limit]  # Ograniczenie liczby przetwarzanych zdjęć

        for img_name in images:
            img_path = os.path.join(subset_dir, img_name)
            img = cv2.imread(img_path)

            if img is None:
                print(f"Pomijam nieprawidłowy obraz: {img_name}")
                continue

            # Przetwarzanie obrazu
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Konwersja do grayscale
            img = cv2.filter2D(img, -1, sharpen_kernel)  # Wyostrzanie obrazu
            img = cv2.resize(img, (64, 64))  # Zmiana rozmiaru
            img = img / 255.0  # Normalizacja do zakresu [0, 1]

            data.append(img)
            labels.append(len(class_names) - 1)  # Indeks aktualnej klasy

    return np.array(data), np.array(labels), class_names

# Wczytanie danych treningowych i testowych
print("Wczytywanie danych treningowych...")
x_train, y_train, class_names = load_data(CROPPED_HANDS_DIR, "train", data_limit)

print("\nWczytywanie danych testowych...")
x_test, y_test, _ = load_data(CROPPED_HANDS_DIR, "test", data_limit)

# Informacje o danych
NUM_CLASSES = len(class_names)
print(f"Liczba klas: {NUM_CLASSES}")
print(f"Liczba zdjęć treningowych: {len(x_train)}")
print(f"Liczba zdjęć testowych: {len(x_test)}")

# Konwersja etykiet do formatu one-hot
y_train = to_categorical(y_train, NUM_CLASSES)
y_test = to_categorical(y_test, NUM_CLASSES)

# Dodanie kanału (1 dla grayscale)
x_train = x_train[..., np.newaxis]
x_test = x_test[..., np.newaxis]

# Definicja modelu CNN
model = Sequential([
    Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(0.01), input_shape=(64, 64, 1)),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(128, (3, 3), activation='relu', kernel_regularizer=l2(0.01)),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(256, (3, 3), activation='relu', kernel_regularizer=l2(0.01)),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),

    Flatten(),
    Dense(512, activation='relu', kernel_regularizer=l2(0.01)),
    Dropout(0.6),
    Dense(256, activation='relu', kernel_regularizer=l2(0.01)),
    Dropout(0.6),
    Dense(NUM_CLASSES, activation='softmax')
])

# Kompilacja modelu
model.compile(
    optimizer=Adam(learning_rate=0.00005),  # Mniejszy learning rate
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Early Stopping, aby zatrzymać trening przy braku poprawy
early_stopping = EarlyStopping(
    monitor='val_accuracy',
    patience=5,  # Liczba epok bez poprawy przed zatrzymaniem
    restore_best_weights=True  # Przywrócenie najlepszych wag
)

# Trenowanie modelu
print("\nTrenowanie modelu...")
history = model.fit(
    x_train, y_train,
    epochs=10,  # Większa liczba epok, ponieważ EarlyStopping zatrzyma uczenie w odpowiednim momencie
    batch_size=32,
    validation_data=(x_test, y_test),
    callbacks=[early_stopping]
)

# Ewaluacja modelu
print("\nEwaluacja na danych testowych:")
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f"Dokładność na danych testowych: {test_accuracy * 100:.2f}%")

# Generowanie macierzy pomyłek
y_pred = model.predict(x_test)  # Przewidywane wartości
y_pred_classes = np.argmax(y_pred, axis=1)  # Konwersja predykcji na indeksy klas
y_true_classes = np.argmax(y_test, axis=1)  # Prawdziwe klasy

cm = confusion_matrix(y_true_classes, y_pred_classes)

# Wizualizacja macierzy pomyłek
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Raport klasyfikacji
print("\nRaport klasyfikacji:")
print(classification_report(y_true_classes, y_pred_classes, target_names=class_names))
