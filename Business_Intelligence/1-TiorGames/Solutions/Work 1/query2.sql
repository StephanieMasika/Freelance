SELECT 
    c.ChampionName,
    COUNT(f.MatchID) AS TotalMatches,
    SUM(f.Wins) AS TotalWins,
    SUM(f.Losses) AS TotalLosses
FROM 
    FactMatchResults f
JOIN 
    DimChampion c ON f.ChampionID = c.ChampionID
GROUP BY 
    c.ChampionName
ORDER BY 
    TotalMatches DESC;
