SELECT B.City, SUM(OD.TotalPrice) AS MonthlyIncome
FROM Orders O
JOIN OrderDetails OD ON O.OrderID = OD.OrderID
JOIN Branch B ON O.BranchID = B.BranchID
WHERE O.OrderDate BETWEEN '2024-07-01' AND '2024-07-31'
GROUP BY B.City;
