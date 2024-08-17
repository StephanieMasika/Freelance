-- Sample Data for Branch
INSERT INTO Branch (BranchName, Address, City) VALUES 
('Waterlooville', '123 Main St', 'Waterlooville'),
('Fareham', '456 Elm St', 'Fareham'),
('Gosport', '789 Oak St', 'Gosport'),
('Havant', '101 Pine St', 'Havant'),
('Chichester', '202 Maple St', 'Chichester'),
('Portsmouth', '303 Birch St', 'Portsmouth');

-- Sample Data for Staff
INSERT INTO Staff (FirstName, LastName, Position, BranchID) VALUES 
('John', 'Doe', 'Manager', 1),
('Jane', 'Smith', 'Cashier', 1),
('Emily', 'Davis', 'Stock Clerk', 2),
('Michael', 'Brown', 'Manager', 3),
('Sarah', 'Wilson', 'Cashier', 3);

-- Sample Data for Product
INSERT INTO Product (ProductName, Category, Price, Ingredients, AllergyAdvice, Lifestyle, SizeVolume, NetWeight, DirectionForUse, NutritionInfo, CountryOfOrigin, StorageInstruction, Manufacturer) VALUES 
('Sliced Malted Bloomer Bread', 'Bread/Bakery', 1.35, 'Wheat Flour, Water, Malted Wheat, Yeast, Rye Flour, Rapeseed Oil, Salt, Wheat Bran, Wheat Semolina, Malted Barley Flour, Preservative: Calcium Propionate, Flour Treatment Agent: Ascorbic Acid', 'Contains: Wheat, Rye, Barley', 'Vegetarian', 'Large', '800g', 'Use as directed', 'Energy 1064kJ, 251kcal, Fat 1.7g, Carbohydrate 48g, Fibre 4.1g, Protein 8.5g, Salt 0.87g', 'UK', 'Store in a cool, dry place', 'BreadCo');

-- Sample Data for Customer
INSERT INTO Customer (FirstName, LastName, Email, Address, City) VALUES 
('Alice', 'Johnson', 'alice.johnson@example.com', '12 Cedar St', 'Waterlooville'),
('Bob', 'Martin', 'bob.martin@example.com', '34 Spruce St', 'Fareham');

-- Sample Data for Orders
INSERT INTO Orders (OrderDate, CustomerID, BranchID) VALUES 
('2024-07-01', 1, 1),
('2024-07-02', 2, 2);

-- Sample Data for OrderDetails
INSERT INTO OrderDetails (OrderID, ProductID, Quantity, TotalPrice) VALUES 
(1, 1, 2, 2.70),
(2, 1, 1, 1.35);
