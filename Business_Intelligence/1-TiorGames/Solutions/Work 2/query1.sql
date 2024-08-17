SELECT
    DATEPART(year, gd.date) AS Year,
    COUNT(DISTINCT pd.player_id) AS TotalPlayers,
    COUNT(DISTINCT CASE WHEN gd.event_type = 'Match' THEN pd.player_id END) AS PlayersInMatches,
    COUNT(DISTINCT CASE WHEN gd.event_type = 'Purchase' THEN pd.player_id END) AS PlayersMakingPurchases
FROM
    game_data gd
JOIN
    player_data pd ON gd.player_id = pd.player_id
GROUP BY
    DATEPART(year, gd.date)
ORDER BY
    Year;
