season = "Winter"
specific_ingredient = "Malted Barley Flour"
description_keyword = "Bread"

query = {
    "Season": season,
    "Ingredients": {"$regex": specific_ingredient, "$options": "i"},
    "ProductName": {"$regex": description_keyword, "$options": "i"}
}

results = products_collection.find(query)
for product in results:
    print(product)
