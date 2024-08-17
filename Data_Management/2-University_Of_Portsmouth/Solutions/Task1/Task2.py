import pymongo
from pymongo import MongoClient
import random

# Connect to MongoDB
client = MongoClient("mongodb://localhost:27017/")

# Create database
db = client['PMM_Grocery_Supermarket']

# Create collection
products_collection = db['products']

# Sample product categories
categories = ["Beverages", "Bread/Bakery", "Canned/Jarred Goods", "Dairy", "Dry/Baking Goods", "Frozen Foods", "Meats", "Produce", "Cleaners", "Personal Care", "Paper Goods"]

# Function to generate random product data
def generate_product_data():
    product_data = []
    for i in range(100):
        product = {
            "ProductID": i + 1,
            "ProductName": f"Product_{i + 1}",
            "Category": random.choice(categories),
            "Price": round(random.uniform(0.5, 50.0), 2),
            "Ingredients": "Sample ingredients list",
            "AllergyAdvice": "Sample allergy advice",
            "Lifestyle": "Sample lifestyle",
            "SizeVolume": "Sample size/volume",
            "NetWeight": f"{random.randint(100, 2000)}g",
            "DirectionForUse": "Sample direction for use",
            "NutritionInfo": "Sample nutrition info",
            "CountryOfOrigin": "Sample country of origin",
            "StorageInstruction": "Sample storage instruction",
            "Manufacturer": "Sample manufacturer"
        }
        product_data.append(product)
    return product_data

# Generate product data
products = generate_product_data()

# Insert data into collection
products_collection.insert_many(products)

print("Database populated with sample products.")

# Sample Queries
print("\nProducts in the 'Beverages' category:")
for product in products_collection.find({"Category": "Beverages"}):
    print(product)

print("\nProducts with price less than £10.00:")
for product in products_collection.find({"Price": {"$lt": 10.0}}):
    print(product)

print("\nCount of products in each category:")
pipeline = [
    {"$group": {"_id": "$Category", "count": {"$sum": 1}}}
]
for result in products_collection.aggregate(pipeline):
    print(result)

print("\nProduct with name 'Product_1':")
product_name = "Product_1"
product = products_collection.find_one({"ProductName": product_name})
print(product)

print("\nProducts with specific ingredient 'Sample ingredients list':")
ingredient = "Sample ingredients list"
for product in products_collection.find({"Ingredients": ingredient}):
    print(product)
