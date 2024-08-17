WITH SalesDiscrepancy AS (
    SELECT
        d.DateYear,
        md.MerchandiseID,
        L.Country,
        SUM(ofs.MerchandiseSold) AS ActualSales,
        SUM(ofs.MerchandiseStocked) AS ExpectedSales,
        SUM(ofs.MerchandiseSold - ofs.MerchandiseStocked) AS SalesDifference
    FROM
        OnlineSalesFact ofs
    JOIN
        DateDim d ON ofs.DateID = d.DateID
    JOIN
        MerchandiseDim md ON ofs.MerchandiseID = md.MerchandiseID
    JOIN
        LocationDim L ON  ofs.MerchandiseID = L.LocationID
    WHERE
        L.Country = 'Japan'
    GROUP BY
        d.DateYear, md.MerchandiseID, L.Country
)
SELECT TOP 1
    s.DateYear,
    s.MerchandiseID,
    s.SalesDifference
FROM
    SalesDiscrepancy s
ORDER BY
    s.SalesDifference DESC;

WITH SalesDiscrepancy AS (
    SELECT
        d.DateYear,
        md.MerchandiseID,
        L.Country,
        SUM(ofs.MerchandiseSold) AS ActualSales,
        SUM(ofs.MerchandiseStocked) AS ExpectedSales,
        SUM(ofs.MerchandiseSold - ofs.MerchandiseStocked) AS SalesDifference
    FROM
        OnlineSalesFact ofs
    JOIN
        DateDim d ON ofs.DateID = d.DateID
    JOIN
        MerchandiseDim md ON ofs.MerchandiseID = md.MerchandiseID
    JOIN
        LocationDim L ON  ofs.MerchandiseID = L.LocationID
    GROUP BY
        d.DateYear, md.MerchandiseID, l.Country
)
SELECT TOP 1
    s.DateYear,
    s.MerchandiseID,
    s.Country,
    s.SalesDifference
FROM
    SalesDiscrepancy s
ORDER BY
    s.SalesDifference ASC;

