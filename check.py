import sqlite3

conn = sqlite3.connect('users.db')
cursor = conn.cursor()

# Execute SQL queries
cursor.execute('SELECT * FROM users')
result = cursor.fetchall()
print(result)

# Close the connection
conn.close()