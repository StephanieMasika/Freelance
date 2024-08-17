SELECT
    dd.region,
    SUM(gd.purchase_amount) AS TotalRevenue
FROM
    game_data gd
JOIN
    player_data pd ON gd.player_id = pd.player_id
JOIN
    demographic_data dd ON pd.demographic_id = dd.demographic_id
WHERE
    gd.event_type = 'Purchase'
GROUP BY
    dd.region
ORDER BY
    TotalRevenue DESC;
