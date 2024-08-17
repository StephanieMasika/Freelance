SELECT
    CASE
        WHEN gd.match_duration < 10 THEN '0-10 mins'
        WHEN gd.match_duration < 20 THEN '10-20 mins'
        WHEN gd.match_duration < 30 THEN '20-30 mins'
        ELSE '30+ mins'
    END AS MatchDurationRange,
    COUNT(gd.game_id) AS TotalMatches
FROM
    game_data gd
WHERE
    gd.event_type = 'Match'
GROUP BY
    CASE
        WHEN gd.match_duration < 10 THEN '0-10 mins'
        WHEN gd.match_duration < 20 THEN '10-20 mins'
        WHEN gd.match_duration < 30 THEN '20-30 mins'
        ELSE '30+ mins'
    END
ORDER BY
    MatchDurationRange;
