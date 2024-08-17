SELECT 
    d.MonthName,
    COUNT(f.PlayerID) AS TotalPlayers,
    SUM(f.MatchDuration) AS TotalPlaytime
FROM 
    FactMatches f
JOIN 
    DimDate d ON f.DateID = d.DateID
GROUP BY 
    d.MonthName
ORDER BY 
    d.MonthName;
