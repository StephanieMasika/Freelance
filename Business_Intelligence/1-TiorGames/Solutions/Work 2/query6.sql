SELECT
    pd.player_id,
    MIN(gd.date) AS FirstActivityDate,
    MAX(gd.date) AS LastActivityDate,
    DATEDIFF(day, MIN(gd.date), MAX(gd.date)) AS DaysActive,
    COUNT(DISTINCT gd.game_id) AS TotalGamesPlayed
FROM
    game_data gd
JOIN
    player_data pd ON gd.player_id = pd.player_id
GROUP BY
    pd.player_id
ORDER BY
    DaysActive DESC;
