import os

def remove_classes_from_labels(labels_path, classes_to_remove):
    """
    Verilen labels klasöründeki tüm .txt dosyalarından belirtilen sınıf indekslerine ait satırları siler.

    :param labels_path: Labels klasörünün yolu (örneğin, 'path/to/your/labels')
    :param classes_to_remove: Silinecek sınıf indekslerinin kümesi (örneğin, {1, 2})
    """
    # Labels klasöründeki tüm .txt dosyalarını işle
    for filename in os.listdir(labels_path):
        if filename.endswith('.txt'):
            filepath = os.path.join(labels_path, filename)
            with open(filepath, 'r') as file:
                lines = file.readlines()  # Dosyadaki tüm satırları oku

            # Yeni satırları oluştur (belirtilen sınıfları hariç tut)
            new_lines = []
            for line in lines:
                class_id = int(line.strip().split()[0])  # Satırdaki sınıf indeksini al
                if class_id not in classes_to_remove:  # Eğer sınıf silinecek sınıflardan değilse
                    new_lines.append(line)  # Yeni satırı ekle

            # Dosyayı güncelle
            with open(filepath, 'w') as file:
                file.writelines(new_lines)

    print(f"{classes_to_remove} numaralı sınıflara ait satırlar silindi!")

def remap_class_indices(labels_path, index_mapping):
    """
    Verilen labels klasöründeki tüm .txt dosyalarındaki sınıf indekslerini yeniden atar.

    :param labels_path: Labels klasörünün yolu (örneğin, 'path/to/your/labels')
    :param index_mapping: Eski indeksleri yeni indekslere eşleyen sözlük (örneğin, {0: 1, 3: 0, 4: 2})
    """
    # Labels klasöründeki tüm .txt dosyalarını işle
    for filename in os.listdir(labels_path):
        if filename.endswith('.txt'):
            filepath = os.path.join(labels_path, filename)
            with open(filepath, 'r') as file:
                lines = file.readlines()  # Dosyadaki tüm satırları oku

            # Yeni satırları oluştur (indeksleri yeniden ata)
            new_lines = []
            for line in lines:
                parts = line.strip().split()  # Satırı boşluklara göre ayır
                class_id = int(parts[0])  # Mevcut sınıf indeksini al

                # Eğer sınıf indeksi index_mapping'de varsa, yeniden ata
                if class_id in index_mapping:
                    parts[0] = str(index_mapping[class_id])  # Yeni indeksi yaz

                # Yeni satırı ekle
                new_lines.append(' '.join(parts) + '\n')

            # Dosyayı güncelle
            with open(filepath, 'w') as file:
                file.writelines(new_lines)

    print("Sınıf indeksleri yeniden atandı!")

#remove_classes_from_labels("helmet_Dataset_Demo/valid/labels", {1, 2})
remap_class_indices("helmet_Dataset_Demo/valid/labels", {0: 1, 3: 0, 4: 2})