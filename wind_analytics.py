import sqlite3

connect = sqlite3.connect("weather.sqlite")
cursor = connect.cursor()

cursor = cursor.execute("select velocity from wind where place == 'kobe';")
cursor