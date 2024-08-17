SELECT City, COUNT(CustomerID) AS CustomerCount
FROM Customer
WHERE CustomerID IN (
SELECT CustomerID 
FROM Orders
WHERE OrderDate BETWEEN '2024-07-01' AND '2024-07-31'
)
GROUP BY City;