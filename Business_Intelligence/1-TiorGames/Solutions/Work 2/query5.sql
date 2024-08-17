SELECT
    YEAR(gd.date) AS Year,
    COUNT(DISTINCT gd.player_id) AS UniquePlayersAttended
FROM
    game_data gd
WHERE
    gd.event_type = 'Event'
GROUP BY
    YEAR(gd.date)
ORDER BY
    Year;
