SELECT B.BranchName, P.ProductName, SUM(OD.Quantity) AS TotalQuantity
FROM Orders O
JOIN OrderDetails OD ON O.OrderID = OD.OrderID
JOIN Product P ON OD.ProductID = P.ProductID
JOIN Branch B ON O.BranchID = B.BranchID
GROUP BY B.BranchName, P.ProductName;
