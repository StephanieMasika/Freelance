SELECT 
    e.EventName,
    COUNT(f.MatchID) AS TotalMatches,
    AVG(f.MatchDuration) AS AvgMatchDuration
FROM 
    FactMatches f
JOIN 
    DimEvent e ON f.EventID = e.EventID
GROUP BY 
    e.EventName
ORDER BY 
    TotalMatches DESC;
