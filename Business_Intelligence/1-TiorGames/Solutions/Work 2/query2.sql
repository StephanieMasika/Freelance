SELECT
    cd.champion_name,
    COUNT(gd.game_id) AS TotalGamesPlayed,
    COUNT(DISTINCT gd.player_id) AS TotalPlayers
FROM
    game_data gd
JOIN
    champion_data cd ON gd.champion_id = cd.champion_id
GROUP BY
    cd.champion_name
ORDER BY
    TotalGamesPlayed DESC;
