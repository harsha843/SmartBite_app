DROP TABLE IF EXISTS users;

CREATE TABLE users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    email TEXT UNIQUE NOT NULL,
    password TEXT NOT NULL,
    age INTEGER NOT NULL,
    gender TEXT NOT NULL,
    height REAL NOT NULL,
    weight REAL NOT NULL,
    activity TEXT NOT NULL,
    face_image_path TEXT,  -- Path to stored image file
    face_image_data TEXT,  -- Base64 encoded image data
    registration_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);