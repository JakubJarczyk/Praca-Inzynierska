import time
import pickle
import os
import random
import mediapipe as mp
import cv2
import matplotlib.pyplot as plt

PHOTOS_DIR = 'photos_datasets'
PICKLE_DIR = 'datasets_pickle'

# Upewniamy się, że katalogi istnieją
os.makedirs(PHOTOS_DIR, exist_ok=True)
os.makedirs(PICKLE_DIR, exist_ok=True)

# Inicjalizacja MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Słowniki do przechowywania statystyk
train_stats = {}
test_stats = {}

# Funkcja do przetwarzania katalogów i zbierania danych
def process_data_set(data_set_path, data_set_type, data, labels, stats, use_z_dimension=False, limit=None):
    """
    Przetwarzanie zdjęć treningowych lub testowych z katalogów znaków.
    """
    valid_extensions = (".jpg", ".jpeg", ".png", ".bmp")

    for class_name in os.listdir(data_set_path):  # Iteracja po znakach
        class_dir = os.path.join(data_set_path, class_name)

        if not os.path.isdir(class_dir):
            print(f"Pomijam {class_name}, nie jest katalogiem klasy.")
            continue

        subset_dir = os.path.join(class_dir, data_set_type)  # train/test
        if not os.path.exists(subset_dir):
            print(f"Katalog {subset_dir} nie istnieje. Pomijam.")
            continue

        print(f"\nPrzetwarzam klasę: {class_name} ({data_set_type})")
        total_images = 0
        detected_images = 0
        rejected_images = 0

        images = [img for img in os.listdir(subset_dir) if img.lower().endswith(valid_extensions)]
        random.shuffle(images)  # Losowe przetasowanie obrazów

        with mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.10) as hands:
            for img_path in images:
                if limit and detected_images >= limit:
                    break

                total_images += 1
                data_aux = []
                img = cv2.imread(os.path.join(subset_dir, img_path))

                if img is None:
                    print(f"Błąd: Nie można załadować obrazu {img_path}. Pomijam.")
                    rejected_images += 1
                    continue

                # Konwersja obrazu do RGB
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # Wykrycie dłoni i landmarków
                results = hands.process(img_rgb)

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        for i in range(len(hand_landmarks.landmark)):
                            x = hand_landmarks.landmark[i].x
                            y = hand_landmarks.landmark[i].y
                            data_aux.append(x)
                            data_aux.append(y)
                            if use_z_dimension:
                                z = hand_landmarks.landmark[i].z
                                data_aux.append(z)

                    data.append(data_aux)
                    labels.append(class_name)
                    detected_images += 1
                    print(f"Przetworzono obraz {img_path} ({detected_images}/{limit or '∞'})")
                else:
                    rejected_images += 1
                    print(f"Odrzucono obraz {img_path}, brak wykrytej dłoni.")

        stats[class_name] = {
            "total_images": total_images,
            "rejected_images": rejected_images,
            "processed_images": detected_images,
            "rejected_percentage": (rejected_images / total_images) * 100 if total_images > 0 else 0
        }

# Wybór zestawu danych
print("Dostępne zestawy danych w katalogu './photos_datasets':")
datasets = os.listdir(PHOTOS_DIR)
for idx, dataset in enumerate(datasets, start=1):
    print(f"{idx}. {dataset}")

try:
    selected_index = int(input("Wybierz numer zestawu danych do przetworzenia: ")) - 1
    if selected_index < 0 or selected_index >= len(datasets):
        raise ValueError("Wybrano nieprawidłowy numer zestawu danych.")
    selected_dataset = datasets[selected_index]
    selected_dataset_path = os.path.join(PHOTOS_DIR, selected_dataset)
except ValueError as e:
    print(f"Błąd: {e}. Kończę program.")
    exit()

# Wybór trybu działania
print("\nWybierz tryb działania:")
print(f"1 - Przetwórz dane treningowe i testowe.")
print(f"2 - Przetwórz dane tylko dla treningu/testu.")
mode = input("Podaj tryb działania (1/2): ")

# Wybór, czy uwzględniać wymiar 'z'
use_z = input("Czy chcesz uwzględniać wymiar 'z' (tak/nie)? ").lower() == 'tak'

# Wybór liczby danych do przetworzenia
use_all = input("Czy chcesz użyć wszystkich zdjęć? (tak/nie): ").lower() == 'tak'
data_limit = None if use_all else int(input("Podaj liczbę zdjęć do przetworzenia: "))

# Inicjalizacja bazy danych
train_data = []
train_labels = []
test_data = []
test_labels = []

if mode == "1":
    print("\nPrzetwarzam dane treningowe...")
    process_data_set(selected_dataset_path, "train", train_data, train_labels, train_stats, use_z_dimension=use_z, limit=data_limit)

    print("\nPrzetwarzam dane testowe...")
    process_data_set(selected_dataset_path, "test", test_data, test_labels, test_stats, use_z_dimension=use_z, limit=data_limit)

elif mode == "2":
    data_set_type = input("Podaj typ danych do przetworzenia (train/test): ").lower()
    stats = train_stats if data_set_type == "train" else test_stats
    if data_set_type in ["train", "test"]:
        process_data_set(selected_dataset_path, data_set_type, train_data if data_set_type == "train" else test_data,
                         train_labels if data_set_type == "train" else test_labels,
                         stats, use_z_dimension=use_z, limit=data_limit)
    else:
        print("Nieprawidłowy typ danych. Kończę program.")
        exit()

else:
    print("Nieprawidłowy tryb działania. Kończę program.")
    exit()

# Zapisanie danych do plików pickle
pickle_name = input("Podaj nazwę pliku pickle (bez rozszerzenia): ")
pickle_train_path = os.path.join(PICKLE_DIR, f"{pickle_name}_train.pickle")
pickle_test_path = os.path.join(PICKLE_DIR, f"{pickle_name}_test.pickle")

with open(pickle_train_path, 'wb') as f:
    pickle.dump({'data': train_data, 'labels': train_labels}, f)
print(f"Zapisano dane treningowe w pliku: {pickle_train_path}")

with open(pickle_test_path, 'wb') as f:
    pickle.dump({'data': test_data, 'labels': test_labels}, f)
print(f"Zapisano dane testowe w pliku: {pickle_test_path}")

# Podsumowanie statystyk dla grupy treningowej i testowej
def print_summary(stats, group_name):
    print(f"\nPodsumowanie przetworzonych zdjęć dla grupy {group_name}:")
    for class_name, stat in stats.items():
        print(f"Kategoria '{class_name}':")
        print(f"  Łączna liczba zdjęć: {stat['total_images']}")
        print(f"  Przetworzone zdjęcia: {stat['processed_images']}")
        print(f"  Odrzucone zdjęcia: {stat['rejected_images']}")
        print(f"  Procent odrzuconych: {stat['rejected_percentage']:.2f}%")

print_summary(train_stats, "treningowej")
print_summary(test_stats, "testowej")
