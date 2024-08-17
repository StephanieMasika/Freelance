SELECT ChampionInGameSpecDim.ChampionItemID
        ,ChampionInGameSpecDim.PlayerInGameID
FROM ChampionInGameSpecDim
INNER JOIN ChampionItemDim ON ChampionInGameSpecDim.ChampionItemID=ChampionItemDim.ChampionItemID
INNER JOIN PlayerInGameDim ON ChampionInGameSpecDim.PlayerInGameID=PlayerInGameDim.PlayerInGameID
WHERE ChampionItemDim.ChampionItemPrice > 89