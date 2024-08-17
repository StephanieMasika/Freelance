SELECT 
    dep.DepartmentName,
    rep.RepresentativeFirstName,
    rep.RepresentativeLastName,
    COUNT(f.QueryResolutionBalance) AS escalated_queries
FROM 
    QueryHandleFact f
JOIN 
    DepartmentDim dep ON f.DepartmentID = dep.DepartmentID
JOIN 
    RepresentativeDim rep ON f.RepresentativeID = rep.RepresentativeID
WHERE 
    f.LevelOfEscalation = 'Yes'
GROUP BY 
    dep.DepartmentName, rep.RepresentativeFirstName, rep.RepresentativeLastName
ORDER BY 
    escalated_queries DESC;
