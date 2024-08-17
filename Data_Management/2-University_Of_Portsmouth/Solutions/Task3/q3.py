delivery_history_query = [
    {
        "$lookup": {
            "from": "order_details",
            "localField": "OrderID",
            "foreignField": "OrderID",
            "as": "order_details"
        }
    },
    {
        "$unwind": "$order_details"
    },
    {
        "$lookup": {
            "from": "products",
            "localField": "order_details.ProductID",
            "foreignField": "ProductID",
            "as": "product_info"
        }
    },
    {
        "$unwind": "$product_info"
    },
    {
        "$project": {
            "OrderID": 1,
            "OrderDate": 1,
            "CustomerID": 1,
            "product_info.ProductName": 1,
            "order_details.Quantity": 1,
            "order_details.TotalPrice": 1
        }
    }
]

results = db.orders.aggregate(delivery_history_query)
for result in results:
    print(result)
