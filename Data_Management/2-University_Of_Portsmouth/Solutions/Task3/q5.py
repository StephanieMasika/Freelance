lifestyle = "Vegetarian"
nutrition_criteria = {"Energy": {"$lt": 500}, "Fat": {"$lt": 5}}

query = {
    "Lifestyle": lifestyle,
    "NutritionInfo": {
        "$elemMatch": {
            "Energy": {"$lt": 500},
            "Fat": {"$lt": 5}
        }
    }
}

results = products_collection.find(query)
for product in results:
    print(product)
