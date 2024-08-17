SELECT
    g.GameStage,
    SUM(pr.PRAssists) AS TotalAssists
FROM
    GameDim g
JOIN
    PlayerInGameDim pg ON g.GameID = pg.GameID
JOIN
    PersonalRecordDim pr ON pg.PRID = pr.PRID
GROUP BY
    g.GameStage;