import os

def rename_jpg_files(folder_path):
    
    # Klasördeki dosyaları al
    files = [f for f in os.listdir(folder_path) if f.endswith('.jpg')]
    files.sort()  # Dosyaları sıralayın (isteğe bağlı)

    # Dosyaları sırayla yeniden adlandır
    for index, file_name in enumerate(files, start=1):
        old_path = os.path.join(folder_path, file_name)
        new_name = f"{index}.jpg"
        new_path = os.path.join(folder_path, new_name)
        os.rename(old_path, new_path)
        print(f"Renamed: {file_name} -> {new_name}")

def rename_txt_files(folder_path):
    
    # Klasördeki dosyaları al
    files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
    files.sort()  # Dosyaları sıralayın (isteğe bağlı)

    # Dosyaları sırayla yeniden adlandır
    for index, file_name in enumerate(files, start=1):
        old_path = os.path.join(folder_path, file_name)
        new_name = f"{index}.txt"
        new_path = os.path.join(folder_path, new_name)
        os.rename(old_path, new_path)
        print(f"Renamed: {file_name} -> {new_name}")


train_img = "helmet_Dataset/train/images"  
test_img = "helmet_Dataset/test/images"  
valid_img = "helmet_Dataset/valid/images"

train_txt= "helmet_Dataset/train/labels"  
test_txt = "helmet_Dataset/test/labels"  
valid_txt = "helmet_Dataset/valid/labels"

rename_jpg_files(train_img)
rename_jpg_files(test_img)
rename_jpg_files(valid_img)

rename_txt_files(train_txt)
rename_txt_files(test_txt)
rename_txt_files(valid_txt)
