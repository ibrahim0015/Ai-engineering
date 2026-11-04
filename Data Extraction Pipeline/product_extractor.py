from pydantic import BaseModel
from google import genai
import os
import sqlite3
import csv
import instructor
import time



class Product(BaseModel):
    title : str
    price : float
    currency : str | None = None
    brand : str | None = None
    category: str | None = None
    available: bool | None = None

class Products(BaseModel):
    products : list[Product]


# Initialize Gemini client
client=instructor.from_genai(
    client = genai.Client(api_key=os.getenv("api_key")),
    mode=instructor.Mode.GENAI_JSON,)

# Instructions
with open ("product_instructions.md", "r") as f:
    instructions = f.read()

# SQLite Initiallization
connection = sqlite3.connect("Products.db")

cursor = connection.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS Products(
id INTEGER PRIMARY KEY,
title TEXT,
price REAL,
currency TEXT,
brand TEXT,
category TEXT,
available BOOLEAN)""")

# Helper
def chunks(lst, size):
    for i in range(0, len(lst), size):
        yield lst[i:i + size]

# Input data
with open("product_inputs.csv", "r", encoding="utf-8") as file:
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

        for product in response.products:
            cursor.execute(
                "SELECT 1 FROM Products WHERE title = ?",
                (product.title,)
            )

            exists = cursor.fetchone()

            if exists:
                print(f"Skipping duplicate: {product.title}")

            else:
                cursor.execute("""
                INSERT INTO Products
                (title, price, currency, brand, category, available)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    product.title,
                    product.price,
                    product.currency,
                    product.brand,
                    product.category,
                    product.available
                ))

        connection.commit()
    except Exception as e:
        print(f"Error: {e}")
        print("Waiting 60 seconds...")
        time.sleep(60)
        


connection.close()





