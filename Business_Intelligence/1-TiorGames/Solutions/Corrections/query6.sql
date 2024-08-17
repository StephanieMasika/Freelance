SELECT *
FROM PlayerInGameDim AS P1
INNER JOIN (
    SELECT *
    FROM PersonalRecordDim
) AS P2 ON P2.PRID = P1.PRID;