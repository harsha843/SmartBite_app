#view_users.py

import sqlite3

conn = sqlite3.connect('nutrition_app.db')
cursor = conn.cursor()



user_id_to_delete = 3  # Replace with the ID or specific condition
cursor.execute("DELETE FROM users WHERE id = ?;", (user_id_to_delete,))
conn.commit()  # Commit the changes
print(f"Data for user with ID {user_id_to_delete} has been deleted.")

# List all tables
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
print("Tables in the database:", cursor.fetchall())

# View all users
cursor.execute("SELECT * FROM users;")
print("\nUsers:")
for row in cursor.fetchall():
    print(row)


conn.close()