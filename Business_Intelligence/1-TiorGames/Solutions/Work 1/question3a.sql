WITH OnlineSales AS (
    SELECT
        M.MerchandiseID,
        M.MerchandiseType,
        M.MerchandiseProviderID,
        O.DateID,
        O.MerchandiseSold,
        O.MerchandiseStocked
    FROM OnlineSalesFact O
    INNER JOIN MerchandiseDim M ON O.MerchandiseID = M.MerchandiseID
    INNER JOIN LocationDim L ON M.MerchandiseProviderID = L.LocationID
    WHERE L.Country = 'Japan'
),
UnexpectedSales AS (
    SELECT
        MerchandiseID,
        DateID,
        (MerchandiseSold - MerchandiseStocked) AS UnexpectedSales
    FROM OnlineSales
)
SELECT TOP 1
    M.MerchandiseID AS OnlineItem,
    D.DateYear AS Year
FROM UnexpectedSales US
INNER JOIN DateDim D ON US.DateID = D.DateID
INNER JOIN MerchandiseDim M ON US.MerchandiseID = M.MerchandiseID
ORDER BY UnexpectedSales DESC;
