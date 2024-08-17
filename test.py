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
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Faces (
            id INTEGER PRIMARY KEY,
            image_path TEXT,
            face_location TEXT,
            face_encoding TEXT
        )
    ''')
    
    conn.commit()
    print("Database setup completed.")  # Debugging output
    return conn

# 3. Save Detected Faces to Database Function
def save_faces_to_db(image_path, face_locations, face_encodings, conn):
    cursor = conn.cursor()
    
    for face_location, face_encoding in zip(face_locations, face_encodings):
        cursor.execute('''
            INSERT INTO Faces (image_path, face_location, face_encoding)
            VALUES (?, ?, ?)
        ''', (image_path, json.dumps(face_location), json.dumps(face_encoding.tolist())))
    
    conn.commit()
    print(f"Saved {len(face_locations)} face(s) from {image_path} to the database.")  # Debugging output

# 4. Process Multiple Images Function
def process_images(directory_path, conn):
    for filename in os.listdir(directory_path):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(directory_path, filename)
            face_locations, face_encodings = detect_faces_and_embeddings(image_path)
            if face_locations:
                save_faces_to_db(image_path, face_locations, face_encodings, conn)
            else:
                print(f"No faces found in {image_path}.")  # Debugging output

# 5. Find and Correlate Similar Faces Function
def correlate_faces(conn, tolerance=0.6):
    cursor = conn.cursor()
    cursor.execute('SELECT id, face_encoding FROM Faces')
    faces = cursor.fetchall()

    encodings = [(face[0], np.array(json.loads(face[1]))) for face in faces]
    
    correlated_faces = []
    
    for i, (id1, encoding1) in enumerate(encodings):
        for id2, encoding2 in encodings[i+1:]:
            distance = np.linalg.norm(encoding1 - encoding2)
            if distance < tolerance:
                correlated_faces.append((id1, id2))
    
    return correlated_faces

# 6. Main Execution
if __name__ == "__main__":
    # Set up the database
    conn = setup_database()
    
    # Path to the directory containing images
    directory_path = 'img'
    
    # Process images in the directory and save detected faces to the database
    process_images(directory_path, conn)
    
    # Correlate similar faces
    correlated_faces = correlate_faces(conn)
    if correlated_faces:
        print("Correlated faces (image IDs):")
        for pair in correlated_faces:
            print(pair)
    else:
        print("No similar faces found.")
    
    # Close the database connection
    conn.close()
