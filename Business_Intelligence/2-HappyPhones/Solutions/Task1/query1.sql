SELECT 
    q.QuestionTopic,
    dt.DateYear,
    dt.DateMonth,
    AVG(qh.QueryResolutionBalance) AS avg_resolution_time
FROM 
    QueryHandleFact qh
JOIN 
    QuestionDim q ON qh.QuestionID = q.QuestionID
JOIN 
    DateTimeDim dt ON qh.DateTimeID = dt.DateTimeID
GROUP BY 
    q.QuestionTopic, dt.DateYear, dt.DateMonth
ORDER BY 
    dt.DateYear, dt.DateMonth, q.QuestionTopic;
