import sqlite3
import os

class WorkersDatabase:
    def __init__(self, db_name, default_table="employees"):
        self.db_name = db_name
        self.default_table = default_table
        #self.create_database(self.default_table)
        #self.create_seed_data(self.default_table)

        if not os.path.exists(self.db_name):  # Eğer veritabanı yoksa oluştur
            self.create_database(self.default_table)
            self.create_seed_data(self.default_table)

    def create_database(self, table_name):

        """Belirtilen tabloyu oluşturur."""
        conn = sqlite3.connect(f"{self.db_name}")
        cursor = conn.cursor()

        cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            surname TEXT NOT NULL,
            age INTEGER NOT NULL
        );
        """)

        conn.commit()
        conn.close()
        print(f"Veritabanı oluşturuldu ve '{table_name}' tablosu hazır.")

    def create_seed_data(self, table_name):
        """Eğer tablo boşsa, örnek çalışan verilerini ekler."""
        conn = sqlite3.connect(f"{self.db_name}")
        cursor = conn.cursor()

        # Tablonun boş olup olmadığını kontrol et
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        count = cursor.fetchone()[0]  

        if count == 0:  # Eğer satır sayısı 0 ise tablo boştur
            employees = [
                ("Emre", "Ozkan", 23),
                ("Kubilay", "Celik", 25),
                ("Zeynep", "Yilmaz", 35)
            ]

            cursor.executemany(f"INSERT INTO {table_name} (name, surname, age) VALUES (?, ?, ?)", employees)
            print(f"Seed Data '{table_name}' tablosuna eklendi.")
        else:
            print(f"'{table_name}' tablosu zaten veri içeriyor, seed data eklenmedi.")

        conn.commit()
        conn.close()

    def list_employees(self, table_name=None):
        """Belirtilen tablodaki tüm çalışanları listeler."""
        if table_name is None:
            table_name = self.default_table

        conn = sqlite3.connect(f"{self.db_name}")
        cursor = conn.cursor()
        
        cursor.execute(f"SELECT * FROM {table_name}")
        employees = cursor.fetchall()
        
        conn.close()
        
        print(f"--- {table_name} Tablosu ---")
        for emp in employees:
            print(emp)
        print("-----------------\n")

    def find_employee(self, name):
        conn = sqlite3.connect(f"{self.db_name}")
        cursor = conn.cursor()
        
        # Büyük/küçük harf duyarsız arama için LOWER fonksiyonunu kullan
        cursor.execute(f"SELECT * FROM {self.default_table} WHERE LOWER(name)=LOWER(?)", (name,))
        employee = cursor.fetchone()
        
        conn.close()
        
        if employee:
            # Sonucu sözlük formatında döndür
            return {
                'name': employee[1],      # name
                'surname': employee[2],    # surname
                'age': employee[3]        # age
            }
        return None

    def add_employee(self, table_name, name, surname, age):
        """Belirtilen tabloya yeni bir çalışan ekler."""
        conn = sqlite3.connect(f"{self.db_name}")
        cursor = conn.cursor()
        
        cursor.execute(f"INSERT INTO {table_name} (name, surname, age) VALUES (?, ?, ?)", (name, surname, age))
        conn.commit()
        conn.close()
        print(f"{name} {surname} '{table_name}' tablosuna eklendi.")

    def delete_employee(self, table_name, emp_id):
        """Belirtilen tabloda verilen ID'ye sahip çalışanı siler."""
        conn = sqlite3.connect(f"{self.db_name}")
        cursor = conn.cursor()
        
        cursor.execute(f"DELETE FROM {table_name} WHERE id=?", (emp_id,))
        conn.commit()
        conn.close()
        print(f"{emp_id} numaralı çalışan '{table_name}' tablosundan silindi.")
