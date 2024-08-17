WITH UnexpectedSales AS (
    SELECT
        os.ItemID,
        os.Country,
        os.Year,
        os.Sales - (SELECT AVG(Sales) FROM OnlineSales WHERE Country = os.Country AND Year = os.Year) AS SalesDifference
    FROM
        OnlineSales os
)
SELECT TOP 1
    od.ItemName,
    os.Country,
    os.Year
FROM
    UnexpectedSales os
JOIN
    OnlineItems oi ON os.ItemID = oi.ItemID
WHERE
    os.SalesDifference = (SELECT MAX(SalesDifference) FROM UnexpectedSales WHERE Country = 'Japan')
ORDER BY
    os.SalesDifference DESC;

SELECT TOP 1
    od.ItemName,
    os.Country,
    os.Year
FROM
    UnexpectedSales os
JOIN
    OnlineItems oi ON os.ItemID = oi.ItemID
WHERE
    os.SalesDifference = (SELECT MIN(SalesDifference) FROM UnexpectedSales)
ORDER BY
    os.SalesDifference ASC;
