SELECT
    e.EventName,
    SUM(ef.GameNumberOfPause) AS TotalPauses,
    SUM(ef.GameInterruption) AS TotalInterruptions
FROM
    GameFact ef
JOIN
    EventDim e ON ef.EventID = e.EventID
GROUP BY
    e.EventName;
