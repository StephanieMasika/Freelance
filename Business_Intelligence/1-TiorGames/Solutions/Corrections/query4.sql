SELECT TOP (30) EventID
        ,TicketsSoldPND
        ,SpectatorsNumber
        ,PromotionCost
        ,PromotionRevenue
FROM EventFact
WHERE SpectatorsNumber > 2500