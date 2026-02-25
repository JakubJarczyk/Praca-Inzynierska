import os
from PIL import Image

def mirror_images(input_folder, output_folder):
    """
    Funkcja pobiera zdjęcia z folderu "PhotosToFlip", odwraca je lustrzanie
    i zapisuje w folderze "output_folder", a następnie usuwa zawartość folderu "PhotosToFlip".

    :param input_folder: Ścieżka do folderu z obrazami wejściowymi
    :param output_folder: Ścieżka do folderu wyjściowego na obrazy odwrócone
    """
    # Tworzenie folderu wyjściowego, jeśli nie istnieje
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iteracja po wszystkich plikach w folderze wejściowym
    for filename in os.listdir(input_folder):
        file_path = os.path.join(input_folder, filename)

        # Sprawdzanie, czy plik jest obrazem
        if os.path.isfile(file_path) and filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif")):
            try:
                # Otwieranie obrazu
                with Image.open(file_path) as img:
                    # Odwracanie obrazu lustrzanie
                    mirrored_img = img.transpose(Image.FLIP_LEFT_RIGHT)

                    # Ścieżka do zapisu nowego obrazu
                    output_path = os.path.join(output_folder, filename)

                    # Zapis obrazu
                    mirrored_img.save(output_path)

                    print(f"Obraz {filename} został odwrócony i zapisany w {output_folder}.")
            except Exception as e:
                print(f"Nie udało się przetworzyć pliku {filename}: {e}")

    # Usuwanie plików z folderu wejściowego
    for filename in os.listdir(input_folder):
        file_path = os.path.join(input_folder, filename)
        try:
            os.remove(file_path)
            print(f"Plik {filename} został usunięty z folderu {input_folder}.")
        except Exception as e:
            print(f"Nie udało się usunąć pliku {filename}: {e}")

if __name__ == "__main__":
    # Ścieżka do folderu wejściowego
    input_folder = "PhotosToFlip"

    # Ścieżka do folderu wyjściowego
    output_folder = "odwrocone_zdjecia"

    # Sprawdzanie, czy folder wejściowy istnieje
    if not os.path.exists(input_folder):
        print(f"Folder {input_folder} nie istnieje. Upewnij się, że utworzyłeś folder z odpowiednimi zdjęciami.")
    else:
        # Wywołanie funkcji
        mirror_images(input_folder, output_folder)
        print(f"Wszystkie obrazy zostały przetworzone i zapisane w folderze '{output_folder}'. Folder '{input_folder}' został wyczyszczony.")
