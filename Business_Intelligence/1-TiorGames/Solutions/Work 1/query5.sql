SELECT 
    c.Country,
    COUNT(DISTINCT p.PlayerID) AS TotalPlayers
FROM 
    DimPlayer p
JOIN 
    DimCountry c ON p.CountryID = c.CountryID
GROUP BY 
    c.Country
ORDER BY 
    TotalPlayers DESC;
