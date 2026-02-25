import os
import cv2
import mediapipe as mp

# Inicjalizacja MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Funkcja do nałożenia landmarków na obraz
def overlay_landmarks(image_path, output_dir="processed_images"):
    # Upewniamy się, że katalog do zapisu istnieje
    os.makedirs(output_dir, exist_ok=True)

    # Wczytanie obrazu
    image = cv2.imread(image_path)
    if image is None:
        print(f"Błąd: Nie można wczytać obrazu {image_path}")
        return

    # Konwersja obrazu na RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Inicjalizacja obiektu MediaPipe Hands
    with mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5) as hands:
        # Przetwarzanie obrazu
        results = hands.process(image_rgb)

        # Jeśli wykryto dłoń, narysuj landmarki
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(128, 128, 128), thickness=1, circle_radius=2),  # Stonowane kolory kropek
                    mp_drawing.DrawingSpec(color=(105, 105, 105), thickness=1, circle_radius=2)   # Stonowane kolory linii
                )
        else:
            print("Brak wykrytej dłoni na obrazie.")

    # Konwersja ścieżki wejściowej na nazwę pliku
    file_name = os.path.basename(image_path)
    output_path = os.path.join(output_dir, f"processed_{file_name}")

    # Zapis obrazu z nałożonymi landmarkami
    cv2.imwrite(output_path, image)
    print(f"Przetworzony obraz zapisano jako: {output_path}")

    # Wyświetlenie obrazu z nałożonymi landmarkami
    cv2.imshow("Landmarki dłoni", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Główna część programu
if __name__ == "__main__":
    image_path = input("Podaj ścieżkę do obrazu dłoni: ")
    overlay_landmarks(image_path)
