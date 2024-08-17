SELECT 
    q.QuestionSeverity,
    dt.DateAmPm,
    AVG(TRY_CAST(qh.CustomerSatisfactionLevel AS FLOAT)) AS avg_satisfaction
FROM 
    QueryHandleFact qh
JOIN 
    QuestionDim q ON qh.QuestionID = q.QuestionID
JOIN 
    DateTimeDim dt ON qh.DateTimeID = dt.DateTimeID
WHERE
    TRY_CAST(qh.CustomerSatisfactionLevel AS FLOAT) IS NOT NULL
GROUP BY 
    q.QuestionSeverity, dt.DateAmPm
ORDER BY 
    q.QuestionSeverity, dt.DateAmPm;
