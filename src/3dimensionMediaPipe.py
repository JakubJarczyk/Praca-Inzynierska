import os
import cv2
import mediapipe as mp

# Inicjalizacja MediaPipe
mp_hands = mp.solutions.hands

PHOTOS_DIR = 'photos_datasets'

def display_landmarks(dir_name, use_z_dimension=False):
    """
    Funkcja wyświetla informacje o landmarkach dłoni wykrytych na obrazach z podanego katalogu.
    """
    class_dir = os.path.join(PHOTOS_DIR, dir_name)

    if not os.path.exists(class_dir):
        print(f"Katalog {class_dir} nie istnieje. Pomijam.")
        return

    valid_extensions = (".jpg", ".jpeg", ".png", ".bmp")

    for sub_dir in os.listdir(class_dir):
        sub_dir_path = os.path.join(class_dir, sub_dir)

        if not os.path.isdir(sub_dir_path):
            print(f"Pomijam {sub_dir}, nie jest katalogiem klasy.")
            continue

        print(f"\nPrzetwarzam klasę: {sub_dir}")

        images = [img for img in os.listdir(sub_dir_path) if img.lower().endswith(valid_extensions)]

        with mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.10) as hands:
            for img_path in images:
                img = cv2.imread(os.path.join(sub_dir_path, img_path))

                if img is None:
                    print(f"Błąd: Nie można załadować obrazu {img_path}. Pomijam.")
                    continue

                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # Wykrycie dłoni i landmarków
                results = hands.process(img_rgb)

                if results.multi_hand_landmarks:
                    print(f"\nObraz: {img_path}")
                    for hand_landmarks in results.multi_hand_landmarks:
                        for i, landmark in enumerate(hand_landmarks.landmark):
                            if use_z_dimension:
                                print(f"Landmark {i}: x={landmark.x:.3f}, y={landmark.y:.3f}, z={landmark.z:.3f}")
                            else:
                                print(f"Landmark {i}: x={landmark.x:.3f}, y={landmark.y:.3f}")
                else:
                    print(f"Brak wykrytej dłoni w obrazie {img_path}.")

# Wybór katalogu do przetwarzania
print("\nDostępne katalogi w './photos_datasets':")
directories = os.listdir(PHOTOS_DIR)
for idx, directory in enumerate(directories, start=1):
    print(f"{idx}. {directory}")

try:
    selected_index = int(input("Wybierz numer katalogu do przetworzenia: ")) - 1
    if selected_index < 0 or selected_index >= len(directories):
        raise ValueError("Wybrano nieprawidłowy numer katalogu.")
    selected_directory = directories[selected_index]
except ValueError as e:
    print(f"Błąd: {e}. Kończę program.")
    exit()

use_z = input("Czy chcesz uwzględniać wymiar 'z' (tak/nie)? ").lower() == 'tak'

print(f"\nPrzetwarzam katalog '{selected_directory}' w './photos_datasets'.")
display_landmarks(selected_directory, use_z_dimension=use_z)
