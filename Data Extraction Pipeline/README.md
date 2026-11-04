# LLM Data Extraction Pipeline

Two Python scripts that use **Gemini API, Instructor, and Pydantic** to extract structured information from unstructured text and store it in **SQLite** databases.

### Projects
- **Customer Data Extraction:** Extracts customer names, phone numbers, emails, products, and reported issues.
- **Product Data Extraction:** Extracts product titles, prices, currencies, brands, categories, and availability. Includes duplicate detection.

### Workflow
1. Read input messages from CSV files.
2. Process messages in batches of 10.
3. Send them to Gemini with custom extraction instructions.
4. Validate structured responses using Pydantic and Instructor.
5. Store extracted data in separate SQLite databases.
6. Handle errors with basic exception handling.

### Requirements
```bash
pip install pydantic google-genai instructor
```

Set your Gemini API key as an environment variable (`api_key`).

### Input Files
- `customer_message_inputs.csv`
- `customer_instructions.md`
- `product_inputs.csv`
- `product_instructions.md`

### Output
- `Customer_issues.db`
- `Products.db`

Run the scripts independently to process customer or product data.