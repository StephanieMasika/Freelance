SELECT
    c.ClubName,
    SUM(ef.MerchandiseSoldPND) AS TotalMerchandiseRevenue
FROM
    EventFact ef
JOIN
    ClubDim c ON ef.EventID = c.ClubID
GROUP BY
    c.ClubName;
