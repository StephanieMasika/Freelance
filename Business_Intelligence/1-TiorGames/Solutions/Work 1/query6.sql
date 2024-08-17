SELECT 
    d.Level,
    COUNT(DISTINCT p.PlayerID) AS TotalPlayers
FROM 
    FactPlayerProgression f
JOIN 
    DimPlayer p ON f.PlayerID = p.PlayerID
JOIN 
    DimLevel d ON f.LevelID = d.LevelID
GROUP BY 
    d.Level
ORDER BY 
    d.Level ASC;
