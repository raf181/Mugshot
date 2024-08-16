import cv2
import face_recognition
import sqlite3
import os
import json

# 1. Face Detection Function
def detect_faces(image_path):
    print(f"Processing image: {image_path}")  # Debugging output
    try:
        # Load the image
        image = cv2.imread(image_path)
        
        # Convert image to RGB (face_recognition uses RGB)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        face_locations = face_recognition.face_locations(rgb_image)
        
        print(f"Found {len(face_locations)} face(s) in {image_path}")  # Debugging output
        return face_locations
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return []

# 2. Database Setup Function
def setup_database():
    conn = sqlite3.connect('faces.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Faces (
            id INTEGER PRIMARY KEY,
            image_path TEXT,
            face_location TEXT
        )
    ''')
    
    conn.commit()
    print("Database setup completed.")  # Debugging output
    return conn

# 3. Save Detected Faces to Database Function
def save_faces_to_db(image_path, face_locations, conn):
    cursor = conn.cursor()
    
    for face_location in face_locations:
        cursor.execute('''
            INSERT INTO Faces (image_path, face_location)
            VALUES (?, ?)
        ''', (image_path, json.dumps(face_location)))
    
    conn.commit()
    print(f"Saved {len(face_locations)} face(s) from {image_path} to the database.")  # Debugging output

# 4. Process Multiple Images Function
def process_images(directory_path, conn):
    for filename in os.listdir(directory_path):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(directory_path, filename)
            face_locations = detect_faces(image_path)
            if face_locations:
                save_faces_to_db(image_path, face_locations, conn)
            else:
                print(f"No faces found in {image_path}.")  # Debugging output

# 5. Query the Database Function
def get_all_faces(conn):
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM Faces')
    return cursor.fetchall()

# 6. Main Execution
if __name__ == "__main__":
    # Set up the database
    conn = setup_database()
    
    # Path to the directory containing images
    directory_path = 'img'
    
    # Process images in the directory and save detected faces to the database
    process_images(directory_path, conn)
    
    # Query and print all detected faces from the database
    faces = get_all_faces(conn)
    if faces:
        print("Detected faces in the following images:")
        for face in faces:
            print(face)
    else:
        print("No faces detected or saved to the database.")
    
    # Close the database connection
    conn.close()
