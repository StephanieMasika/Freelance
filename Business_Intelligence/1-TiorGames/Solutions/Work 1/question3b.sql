WITH OnlineSales AS (
    SELECT
        M.MerchandiseID,
        M.MerchandiseType,
        L.Country,
        D.DateYear,
        (O.MerchandiseSold - O.MerchandiseStocked) AS UnexpectedSales
    FROM OnlineSalesFact O
    INNER JOIN MerchandiseDim M ON O.MerchandiseID = M.MerchandiseID
    INNER JOIN LocationDim L ON M.MerchandiseProviderID = L.LocationID
    INNER JOIN DateDim D ON O.DateID = D.DateID
),
OverallUnexpectedSales AS (
    SELECT
        MerchandiseID,
        Country,
        DateYear,
        SUM(UnexpectedSales) AS TotalUnexpectedSales
    FROM OnlineSales
    GROUP BY MerchandiseID, Country, DateYear
)
SELECT TOP 1
    MerchandiseID AS OnlineItem,
    Country,
    DateYear AS Year
FROM OverallUnexpectedSales
ORDER BY TotalUnexpectedSales ASC;

