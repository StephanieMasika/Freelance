SELECT
    r.RefundType,
    COUNT(r.RefundID) AS TotalRefunds
FROM
    RefundFact rf
JOIN
    RefundDim r ON rf.RefundID = r.RefundID
GROUP BY
    r.RefundType;
