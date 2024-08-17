SELECT
    e.EventName,
    SUM(ef.TicketsSold) AS TotalTicketsSold
FROM
    EventFact ef
JOIN
    EventDim e ON ef.EventID = e.EventID
GROUP BY
    e.EventName;