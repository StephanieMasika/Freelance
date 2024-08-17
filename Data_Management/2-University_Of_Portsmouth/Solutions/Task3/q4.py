purchasing_record_query = [
    {
        "$lookup": {
            "from": "orders",
            "localField": "OrderID",
            "foreignField": "OrderID",
            "as": "order_info"
        }
    },
    {
        "$unwind": "$order_info"
    },
    {
        "$lookup": {
            "from": "branches",
            "localField": "order_info.BranchID",
            "foreignField": "BranchID",
            "as": "branch_info"
        }
    },
    {
        "$unwind": "$branch_info"
    },
    {
        "$project": {
            "ProductName": 1,
            "branch_info.BranchName": 1,
            "branch_info.City": 1,
            "order_info.OrderDate": 1,
            "order_info.CustomerID": 1,
            "Quantity": 1,
            "TotalPrice": 1
        }
    },
    {
        "$sort": {"branch_info.City": 1, "order_info.OrderDate": 1}
    }
]

results = products_collection.aggregate(purchasing_record_query)
for result in results:
    print(result)
