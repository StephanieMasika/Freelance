CREATE TABLE Branch (
    BranchID SERIAL PRIMARY KEY,
    BranchName VARCHAR(100),
    Address VARCHAR(255),
    City VARCHAR(100)
);

CREATE TABLE Staff (
    StaffID SERIAL PRIMARY KEY,
    FirstName VARCHAR(100),
    LastName VARCHAR(100),
    Position VARCHAR(100),
    BranchID INT,
    FOREIGN KEY (BranchID) REFERENCES Branch(BranchID)
);

CREATE TABLE Product (
    ProductID SERIAL PRIMARY KEY,
    ProductName VARCHAR(100),
    Category VARCHAR(100),
    Price DECIMAL(10, 2),
    Ingredients TEXT,
    AllergyAdvice TEXT,
    Lifestyle VARCHAR(100),
    SizeVolume VARCHAR(100),
    NetWeight VARCHAR(100),
    DirectionForUse TEXT,
    NutritionInfo TEXT,
    CountryOfOrigin VARCHAR(100),
    StorageInstruction TEXT,
    Manufacturer VARCHAR(100)
);

CREATE TABLE Customer (
    CustomerID SERIAL PRIMARY KEY,
    FirstName VARCHAR(100),
    LastName VARCHAR(100),
    Email VARCHAR(100),
    Address VARCHAR(255),
    City VARCHAR(100)
);

CREATE TABLE Orders (
    OrderID SERIAL PRIMARY KEY,
    OrderDate DATE,
    CustomerID INT,
    BranchID INT,
    FOREIGN KEY (CustomerID) REFERENCES Customer(CustomerID),
    FOREIGN KEY (BranchID) REFERENCES Branch(BranchID)
);

CREATE TABLE OrderDetails (
    OrderDetailID SERIAL PRIMARY KEY,
    OrderID INT,
    ProductID INT,
    Quantity INT,
    TotalPrice DECIMAL(10, 2),
    FOREIGN KEY (OrderID) REFERENCES Orders(OrderID),
    FOREIGN KEY (ProductID) REFERENCES Product(ProductID)
);
