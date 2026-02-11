
import sqlite3
import numpy as np
import io
import time

class Database:
    def __init__(self, db_path="office_productivity.db"):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self.create_table()

    def create_table(self):
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS persons (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                embedding BLOB,
                status INTEGER DEFAULT 0, -- 0: OUT, 1: IN
                entry_time REAL
            )
        ''')
        self.conn.commit()

    @staticmethod
    def adapt_array(arr):
        """Convert numpy array to binary blob."""
        out = io.BytesIO()
        np.save(out, arr)
        out.seek(0)
        return sqlite3.Binary(out.read())

    @staticmethod
    def convert_array(text):
        """Convert binary blob to numpy array."""
        out = io.BytesIO(text)
        out.seek(0)
        return np.load(out)

    def add_person(self, name, embedding):
        """Add a new person and mark as IN."""
        embedding_blob = self.adapt_array(embedding)
        current_time = time.time()
        self.cursor.execute('INSERT INTO persons (name, embedding, status, entry_time) VALUES (?, ?, ?, ?)',
                            (name, embedding_blob, 1, current_time))
        self.conn.commit()
        return self.cursor.lastrowid

    def update_status(self, person_id, status):
        """Update IN/OUT status."""
        current_time = time.time()
        if status == 1:
            self.cursor.execute('UPDATE persons SET status = ?, entry_time = ? WHERE id = ?', (status, current_time, person_id))
        else:
             self.cursor.execute('UPDATE persons SET status = ? WHERE id = ?', (status, person_id))
        self.conn.commit()

    def get_all_embeddings(self):
        """Retrieve all embeddings and IDs."""
        self.cursor.execute('SELECT id, name, embedding, status, entry_time FROM persons')
        rows = self.cursor.fetchall()
        results = []
        for row in rows:
            results.append({
                'id': row[0],
                'name': row[1],
                'embedding': self.convert_array(row[2]),
                'status': row[3],
                'entry_time': row[4]
            })
        return results

    def get_person(self, person_id):
         self.cursor.execute('SELECT id, name, status, entry_time FROM persons WHERE id = ?', (person_id,))
         return self.cursor.fetchone()

    def close(self):
        self.conn.close()
