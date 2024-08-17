SELECT PlayerInGameDim.PlayerID
FROM PlayerInGameDim
GROUP BY PlayerInGameDim.PlayerID
HAVING COUNT(*) > 5