from pydantic import BaseModel
from google import genai
import os
import sqlite3
import csv
import instructor
import time



class Product(BaseModel):
    name : str
    phone_number : str
    email : str | None = None
    product : str | None = None
    issue : str | None = None
    

class Products(BaseModel):
    products : list[Product]


# Initialize Gemini client
client=instructor.from_genai(
    client = genai.Client(api_key=os.getenv("api_key")),
    mode=instructor.Mode.GENAI_JSON,)

# Instructions
with open ("customer_instructions.md", "r") as f:
    instructions = f.read()

# SQLite Initiallization
connection = sqlite3.connect("Customer_issues.db")

cursor = connection.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS Products(
id INTEGER PRIMARY KEY,
name TEXT,
phone_number TEXT,
email TEXT,
product TEXT,
issue TEXT
)""")

# Helper
def chunks(lst, size):
    for i in range(0, len(lst), size):
        yield lst[i:i + size]

# Input data
with open("customer_message_inputs.csv", "r", encoding="utf-8") as file:
    reader = csv.DictReader(file)

    texts = [row["text"] for row in reader]



for batch in chunks(texts, 10):

    prompt = "\n".join(
        f"{i+1}. {text}"
        for i, text in enumerate(batch)
    )
    try:
        response = client.chat.completions.create(
            model="gemini-3.5-flash-lite",
            response_model=Products,
            messages=[
                {
                    "role": "system",
                    "content": instructions
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        for customer in response.products:

            cursor.execute("""
            INSERT INTO Products
            (name, phone_number, email, product, issue)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                customer.name,
                customer.phone_number,
                customer.email,
                customer.product,
                customer.issue,
            ))
        

        connection.commit()
    except Exception as e:
        print(f"Error: {e}")
        print("Waiting 60 seconds...")
        time.sleep(60)
        


connection.close()





