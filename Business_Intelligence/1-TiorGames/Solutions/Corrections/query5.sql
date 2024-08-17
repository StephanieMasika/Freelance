SELECT PlayerID
        ,PlayerOriginID
        ,PlayerRealName
        ,ClubDim.ClubLocation
FROM PlayerDim
FULL JOIN ClubDim ON PlayerDim.PlayerOriginID=ClubDim.ClubLocation