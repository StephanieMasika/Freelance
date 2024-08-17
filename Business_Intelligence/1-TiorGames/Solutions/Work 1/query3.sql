SELECT 
    p.PlayerID,
    p.PlayerName,
    SUM(f.PurchaseAmount) AS TotalSpent
FROM 
    FactPurchases f
JOIN 
    DimPlayer p ON f.PlayerID = p.PlayerID
GROUP BY 
    p.PlayerID, p.PlayerName
ORDER BY 
    TotalSpent DESC;
