SELECT O.OrderID, O.OrderDate, C.FirstName, C.LastName, C.Address, C.City, B.BranchName, P.ProductName, OD.Quantity, OD.TotalPrice
FROM Orders O
JOIN Customer C ON O.CustomerID = C.CustomerID
JOIN Branch B ON O.BranchID = B.BranchID
JOIN OrderDetails OD ON O.OrderID = OD.OrderID
JOIN Product P ON OD.ProductID = P.ProductID;
