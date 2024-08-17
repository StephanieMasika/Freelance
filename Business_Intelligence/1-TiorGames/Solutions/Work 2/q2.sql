SELECT
    p.PromotionType,
    AVG(p.PromotionDuration) AS AvgPromotionDuration
FROM
    PromotionDim p
GROUP BY
    p.PromotionType;
