
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
                name TEXT UNIQUE,
                status INTEGER DEFAULT 0, -- 0: OUT, 1: IN
                entry_time REAL
            )
        ''')
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id INTEGER,
                embedding BLOB,
                FOREIGN KEY (person_id) REFERENCES persons(id) ON DELETE CASCADE
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

    def add_person(self, name, embedding=None):
        """Add a new person and optionally their first embedding."""
        current_time = time.time()
        try:
            self.cursor.execute('INSERT INTO persons (name, status, entry_time) VALUES (?, ?, ?)',
                                (name, 1, current_time))
            person_id = self.cursor.lastrowid
            
            if embedding is not None:
                self.add_embedding(person_id, embedding)
            
            self.conn.commit()
            return person_id
        except sqlite3.IntegrityError:
            # Name already exists
            self.cursor.execute('SELECT id FROM persons WHERE name = ?', (name,))
            return self.cursor.fetchone()[0]

    def add_embedding(self, person_id, embedding):
        """Add an embedding for an existing person."""
        embedding_blob = self.adapt_array(embedding)
        self.cursor.execute('INSERT INTO embeddings (person_id, embedding) VALUES (?, ?)',
                            (person_id, embedding_blob))
        self.conn.commit()

    def update_status(self, person_id, status):
        """Update IN/OUT status."""
        current_time = time.time()
        if status == 1:
            self.cursor.execute('UPDATE persons SET status = ?, entry_time = ? WHERE id = ?', (status, current_time, person_id))
        else:
             self.cursor.execute('UPDATE persons SET status = ? WHERE id = ?', (status, person_id))
        self.conn.commit()

    def get_all_embeddings(self):
        """Retrieve all persons and their associated embeddings."""
        self.cursor.execute('SELECT id, name, status, entry_time FROM persons')
        person_rows = self.cursor.fetchall()
        
        results = []
        for p_row in person_rows:
            p_id = p_row[0]
            self.cursor.execute('SELECT embedding FROM embeddings WHERE person_id = ?', (p_id,))
            emb_rows = self.cursor.fetchall()
            
            embeddings = [self.convert_array(r[0]) for r in emb_rows]
            
            results.append({
                'id': p_id,
                'name': p_row[1],
                'embeddings': embeddings, # List of arrays
                'status': p_row[2],
                'entry_time': p_row[3]
            })
        return results

    def get_person(self, person_id):
         self.cursor.execute('SELECT id, name, status, entry_time FROM persons WHERE id = ?', (person_id,))
         return self.cursor.fetchone()

    def close(self):
        self.conn.close()
