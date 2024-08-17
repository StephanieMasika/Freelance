availability_query = [
    {
        "$lookup": {
            "from": "branches",
            "localField": "BranchID",
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
            "branch_info.Address": 1,
            "branch_info.City": 1,
            "Quantity": 1
        }
    }
]

results = products_collection.aggregate(availability_query)
for result in results:
    print(result)
