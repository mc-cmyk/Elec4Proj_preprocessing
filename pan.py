import pandas as pd
from sqlalchemy import create_engine

USERNAME = 'root'
PASSWORD = ''
HOST = 'localhost'
PORT = '3306'
DATABASE = 'company_db'

connection_string = f"mysql+pymysql://{USERNAME}:{PASSWORD}@{HOST}:{PORT}/{DATABASE}"
engine = create_engine(connection_string)


query = 'SELECT * FROM employee;'
df = pd.read_sql(query, con=engine)


print("1. Display Column Names & Display Data Types")
print("Columns: ", df.columns.to_list(), "\n")
print("DataTypes: ", df.dtypes)
print("\n")


rows, cols = df.shape
print(" 2. Display Rows, Columns")
print(f"Rows: {rows}, Columns: {cols}\n")


print(" 3. Display dataset structure")
df.info()
print("\n")


print("4. Generate summary statistics")
print(df.describe())
print("\n")

print("5. View first 5 rows")
print(df.head(5))
print("\n")


print("6. View last 5 rows")
print(df.tail(5))
print("\n")


print("7. View random records of 5 ")
print(df.sample(5))
print("\n")

print("8. Filter employees with a salary greater than 55,000.")
stonks = df[df['salary'] > 55000]
print(stonks, "\n")


print("9. Count the number of employees per department.")
dept_count = df['department'].value_counts()
print(dept_count, "\n")


print("10. Sort employees by hire date (ascending).")
df['hire_date'] = pd.to_datetime(df['hire_date'])
hire_sorted = df.sort_values(by='hire_date', ascending=True)
print(hire_sorted, "\n")
