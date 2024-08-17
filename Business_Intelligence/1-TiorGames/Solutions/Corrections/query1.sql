SELECT PlayerInGameDim.PlayerID
        ,PlayerInGameDim.ChampionID
        ,PlayerInGameDim.PlayerInGamePositionInGame
        ,PersonalRecordDim.PRID
FROM PlayerInGameDim
INNER JOIN PersonalRecordDim ON PlayerInGameDim.PRID=PersonalRecordDim.PRID
WHERE PRKills > 2

