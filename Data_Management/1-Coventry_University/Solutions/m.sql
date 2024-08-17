CREATE TABLE Department (
    deptId NUMBER PRIMARY KEY,
    name VARCHAR2(100)
);

CREATE TABLE SalaryGrade (
    salaryCode VARCHAR2(10) PRIMARY KEY,
    startSalary NUMBER,
    finishSalary NUMBER
);

CREATE TABLE PensionScheme (
    schemeId NUMBER PRIMARY KEY,
    name VARCHAR2(100),
    rate NUMBER
);

CREATE TABLE Employee (
    empId NUMBER PRIMARY KEY,
    name VARCHAR2(100),
    address VARCHAR2(200),
    DOB DATE,
    job VARCHAR2(100),
    salaryCode VARCHAR2(10),
    deptId NUMBER,
    manager NUMBER,
    schemeId NUMBER,
    FOREIGN KEY (deptId) REFERENCES Department(deptId),
    FOREIGN KEY (salaryCode) REFERENCES SalaryGrade(salaryCode),
    FOREIGN KEY (schemeId) REFERENCES PensionScheme(schemeId)
);
