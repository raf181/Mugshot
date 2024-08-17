import cv2
import face_recognition
import sqlite3
import os
import json
import numpy as np

# 1. Face Detection and Embedding Function
def detect_faces_and_embeddings(image_path):
    print(f"Processing image: {image_path}")  # Debugging output
    try:
        # Load the image
        image = cv2.imread(image_path)
        
        # Convert image to RGB (face_recognition uses RGB)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect face locations and embeddings
        face_locations = face_recognition.face_locations(rgb_image)
        face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
        
        print(f"Found {len(face_locations)} face(s) in {image_path}")  # Debugging output
        return face_locations, face_encodings
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return [], []

# 2. Database Setup Function
def setup_database():
    conn = sqlite3.connect('faces.db')
    cursor = conn.cursor()
    
    # Table to store unique faces
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS FaceIDs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            face_encoding TEXT UNIQUE
        )
    ''')
    
    # Table to link faces with images
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS FaceImages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            face_id INTEGER,
            image_path TEXT,
            face_location TEXT,
            FOREIGN KEY (face_id) REFERENCES FaceIDs(id)
        )
    ''')
    
    conn.commit()
    print("Database setup completed.")  # Debugging output
    return conn

# 3. Function to Find or Create a Face ID
def get_or_create_face_id(face_encoding, conn, tolerance=0.6):
    cursor = conn.cursor()
    cursor.execute('SELECT id, face_encoding FROM FaceIDs')
    existing_faces = cursor.fetchall()
    
    for face_id, db_encoding in existing_faces:
        db_encoding_array = np.array(json.loads(db_encoding))
        distance = np.linalg.norm(face_encoding - db_encoding_array)
        if distance < tolerance:
            print(f"Matched with existing face ID: {face_id}")
            return face_id
    
    # If no match found, create a new face ID
    cursor.execute('''
        INSERT INTO FaceIDs (face_encoding)
        VALUES (?)
    ''', (json.dumps(face_encoding.tolist()),))
    
    conn.commit()
    new_face_id = cursor.lastrowid
    print(f"Created new face ID: {new_face_id}")
    return new_face_id

# 4. Save Detected Faces to Database Function
def save_faces_to_db(image_path, face_locations, face_encodings, conn):
    for face_location, face_encoding in zip(face_locations, face_encodings):
        face_id = get_or_create_face_id(face_encoding, conn)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO FaceImages (face_id, image_path, face_location)
            VALUES (?, ?, ?)
        ''', (face_id, image_path, json.dumps(face_location)))
    
    conn.commit()
    print(f"Saved {len(face_locations)} face(s) from {image_path} to the database.")  # Debugging output

# 5. Process Multiple Images Function
def process_images(directory_path, conn):
    for filename in os.listdir(directory_path):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(directory_path, filename)
            face_locations, face_encodings = detect_faces_and_embeddings(image_path)
            if face_locations:
                save_faces_to_db(image_path, face_locations, face_encodings, conn)
            else:
                print(f"No faces found in {image_path}.")  # Debugging output

# 6. Query Images by Face ID Function
def find_images_by_face_id(face_id, conn):
    cursor = conn.cursor()
    cursor.execute('''
        SELECT image_path, face_location FROM FaceImages WHERE face_id = ?
    ''', (face_id,))
    
    return cursor.fetchall()

# 7. Main Execution
if __name__ == "__main__":
    # Set up the database
    conn = setup_database()
    
    # Path to the directory containing images
    directory_path = 'img'
    
    # Process images in the directory and save detected faces to the database
    process_images(directory_path, conn)
    
    # Example: Query images by a specific face ID
    cursor = conn.cursor()
    cursor.execute('SELECT id FROM FaceIDs')
    face_ids = cursor.fetchall()

    for face_id_tuple in face_ids:
        face_id = face_id_tuple[0]
        images = find_images_by_face_id(face_id, conn)
        if images:
            print(f"Images containing face ID {face_id}:")
            for image in images:
                print(image)
        else:
            print(f"No images found for face ID {face_id}.")
    
    # Close the database connection
    conn.close()
