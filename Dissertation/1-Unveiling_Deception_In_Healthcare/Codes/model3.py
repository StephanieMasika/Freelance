import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
# importing neigborssum from sklearn
from sklearn import metrics
from sklearn import model_selection
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report, accuracy_score  
from sklearn.metrics import precision_score, recall_score 
from sklearn.metrics import f1_score, matthews_corrcoef 
from sklearn.metrics import confusion_matrix 

# %matplotlib inline

# Building the Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier

# Building the Support Vector Machine
from sklearn import svm

# Building the KNeighbors Classifier
from sklearn.neighbors import KNeighborsClassifier

import os
for dirname, _, filenames in os.walk(r'E:\Personal Home\Work\Freelancing\Freelance\Dissertation\1-Unveiling_Deception_In_Healthcare\Datasets\KAGGLE'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# READING THE DATA
# beneficiary data
beneficiary_train = pd.read_csv(r'E:\Personal Home\Work\Freelancing\Freelance\Dissertation\1-Unveiling_Deception_In_Healthcare\Datasets\KAGGLE\Train_Beneficiarydata-1542865627584.csv')
beneficiary_test= pd.read_csv(r'E:\Personal Home\Work\Freelancing\Freelance\Dissertation\1-Unveiling_Deception_In_Healthcare\Datasets\KAGGLE\Test_Beneficiarydata-1542969243754.csv')

# inpatient data
inpatient_test = pd.read_csv(r'E:\Personal Home\Work\Freelancing\Freelance\Dissertation\1-Unveiling_Deception_In_Healthcare\Datasets\KAGGLE\Test_Inpatientdata-1542969243754.csv')
inpatient_train = pd.read_csv(r'E:\Personal Home\Work\Freelancing\Freelance\Dissertation\1-Unveiling_Deception_In_Healthcare\Datasets\KAGGLE\Train_Inpatientdata-1542865627584.csv')

# outpatient data
outpatient_test= pd.read_csv(r'E:\Personal Home\Work\Freelancing\Freelance\Dissertation\1-Unveiling_Deception_In_Healthcare\Datasets\KAGGLE\Test_Outpatientdata-1542969243754.csv')
outpatient_train = pd.read_csv(r'E:\Personal Home\Work\Freelancing\Freelance\Dissertation\1-Unveiling_Deception_In_Healthcare\Datasets\KAGGLE\Train_Outpatientdata-1542865627584.csv')

#label data
label_train = pd.read_csv(r'E:\Personal Home\Work\Freelancing\Freelance\Dissertation\1-Unveiling_Deception_In_Healthcare\Datasets\KAGGLE\Train-1542865627584.csv')
label_test = pd.read_csv(r'E:\Personal Home\Work\Freelancing\Freelance\Dissertation\1-Unveiling_Deception_In_Healthcare\Datasets\KAGGLE\Test-1542969243754.csv')

# BENEFICIARY DATA
beneficiary_train.info()

# Beneficiary data EDA - Training set
beneficiary_train.shape, beneficiary_test.shape

beneficiary_train.columns

# Gender Distribution of Data
def func(pct, allvalues):
    absolute = int(pct / 100. * np.sum(allvalues))
    return "{:.1f}% ({:d})".format(pct, absolute)

plt.figure(figsize=(8, 6), dpi=100)

data = beneficiary_train.groupby('Gender').count().BeneID

# Updated portion for pie chart creation
plt.pie(data, 
        labels=data.index, 
        autopct=lambda pct: func(pct, data.values),
        startangle=140)  # Adding a start angle for better visualization

plt.title("Gender Distribution Among Beneficiary Data")
plt.show()

# Cleaning the Data
beneficiary_train['RenalDiseaseIndicator'][:10]

# this value indicates whether the beneficiary has renal disease
# Convert 'RenalDiseaseIndicator' to numeric by replacing 'Y' with '1' and converting to int
beneficiary_train['RenalDiseaseIndicator'].replace('Y', '1', inplace=True)
beneficiary_train['RenalDiseaseIndicator'] = beneficiary_train['RenalDiseaseIndicator'].astype(int)

beneficiary_test['RenalDiseaseIndicator'].replace('Y', '1', inplace=True)
beneficiary_test['RenalDiseaseIndicator'] = beneficiary_test['RenalDiseaseIndicator'].astype(int)

# Convert 'DOB' and 'DOD' to datetime
for col in ['DOB', 'DOD']:
    beneficiary_train[col] = pd.to_datetime(beneficiary_train[col])
    beneficiary_test[col] = pd.to_datetime(beneficiary_test[col])

# Extract 'BirthYear' from 'DOB'
beneficiary_train['BirthYear'] = beneficiary_train['DOB'].dt.year
beneficiary_test['BirthYear'] = beneficiary_test['DOB'].dt.year

# Fill NaN 'DOD' values with the greatest Date of Death in the dataset (excluding NaN)
max_bene_DOD_train = beneficiary_train['DOD'].max()
beneficiary_train['DOD'].fillna(value=max_bene_DOD_train, inplace=True)

max_bene_DOD_test = beneficiary_test['DOD'].max()
beneficiary_test['DOD'].fillna(value=max_bene_DOD_test, inplace=True)

# Calculate Age at the time of death
beneficiary_train['Age'] = ((beneficiary_train['DOD'] - beneficiary_train['DOB']).dt.days / 365).round(0)
beneficiary_test['Age'] = ((beneficiary_test['DOD'] - beneficiary_test['DOB']).dt.days / 365).round(0)

# Determine if the beneficiary is alive (NaN 'DOD' means alive)
beneficiary_train['Alive'] = beneficiary_train['DOD'].isna().astype(int)
beneficiary_test['Alive'] = beneficiary_test['DOD'].isna().astype(int)

# Plot the distribution of birth years for beneficiaries, separated by 'Alive' status
fig, ax = plt.subplots(figsize=(15, 6), dpi=100)
sns.histplot(data=beneficiary_train, x='BirthYear', hue='Alive', legend=False, ax=ax)
plt.legend(['Alive', 'Dead'])
plt.title('Distribution of Birth Years of Beneficiaries')
plt.tight_layout()
plt.show()

# Plot the distribution between Alive and Not
plt.figure(figsize=(8, 6))
alive_counts = beneficiary_train['Alive'].value_counts()
bar = plt.bar(alive_counts.index, height=alive_counts.values)
plt.title('Distribution Between Alive and Deceased')
plt.xticks([0, 1], ['Dead', 'Alive'])

# Add annotations to the bar chart
labels = ['Dead', 'Alive']
for i, p in enumerate(bar):
    height = p.get_height()
    plt.annotate(f'{labels[i]}', (p.get_x() + p.get_width() / 2, height + 200), ha='center')
    plt.annotate(f'{(height / beneficiary_train["Alive"].shape[0] * 100):.2f}%', 
                 (p.get_x() + p.get_width() / 2, height + 400), ha='center')

plt.show()

beneficiary_train.drop(labels=['DOD','BirthYear'],axis=1,inplace=True)
beneficiary_test.drop(labels=['DOD','BirthYear'],axis=1,inplace=True)

beneficiary_train.groupby('NoOfMonths_PartACov').count()
beneficiary_train.groupby('NoOfMonths_PartBCov').count()

diseases = ['ChronicCond_Alzheimer','ChronicCond_Heartfailure',
           'ChronicCond_KidneyDisease','ChronicCond_Cancer','ChronicCond_ObstrPulmonary',
          'ChronicCond_Depression','ChronicCond_Diabetes','ChronicCond_IschemicHeart','ChronicCond_Osteoporasis',
           'ChronicCond_rheumatoidarthritis','ChronicCond_stroke']

df_train = beneficiary_train.copy()
df_test = beneficiary_test.copy()

df_train.shape
df_test.shape

# Function to calculate ChronicDiseaseIndex for a single row
def calculate_chronic_disease_count(row, diseases):
    return sum(row[disease] > 1 for disease in diseases)

# Apply the function to both train and test datasets
for df in [df_train, df_test]:
    df['ChronicDiseaseIndex'] = df.apply(calculate_chronic_disease_count, axis=1, diseases=diseases)


df_train.drop(diseases,inplace=True,axis=1)
df_test.drop(diseases,inplace=True,axis=1)

df_train

fig, ax = plt.subplots(figsize=(10,4),dpi=100)
data = df_train.groupby('ChronicDiseaseIndex').count().BeneID
plt.title('the dsitribution of ChronicDisease Index')
plt.bar(x=data.index,height=data,color='navy')
plt.xticks(np.arange(0,12,1))
plt.show()

beneficiary_train = df_train.copy()
beneficiary_test = df_train.copy()

for df in [beneficiary_train,beneficiary_test]:
    for col in ['Race','State','County']:
        df[col] = LabelEncoder().fit_transform(df[col])

beneficiary_train.columns

#INPATIENT AND OUTPATIENT DATA
inpatient_train.head()

# Convert specified date columns to datetime and compute the claim period
def process_claim_dates(df, date_columns):
    for col in date_columns:
        df[col] = pd.to_datetime(df[col])
    df['ClaimPeriod'] = (df['ClaimEndDt'] - df['ClaimStartDt']).dt.days.round(0)

# Convert admission and discharge dates to datetime, compute time in hospital, and drop original columns
def process_hospital_stay(df):
    df['AdmissionDt'] = pd.to_datetime(df['AdmissionDt'])
    df['DischargeDt'] = pd.to_datetime(df['DischargeDt'])
    df['TimeInHptal'] = (df['DischargeDt'] - df['AdmissionDt']).dt.days.round(0)
    df.drop(['DischargeDt', 'AdmissionDt'], axis=1, inplace=True)

# List of dataframes and columns to process
claim_dataframes = [inpatient_train, inpatient_test, outpatient_test, outpatient_train]
hospital_dataframes = [inpatient_train, inpatient_test]

# Process claim dates for all dataframes
for df in claim_dataframes:
    process_claim_dates(df, ['ClaimStartDt', 'ClaimEndDt'])

# Process hospital stay dates for inpatient dataframes
for df in hospital_dataframes:
    process_hospital_stay(df)

data = inpatient_train.groupby('TimeInHptal').count().BeneID
plt.title("the dsitribution of inpatient's time in Hospital")
plt.bar(x=data.index,height=data,color='navy')
plt.xticks(np.arange(0,12,1))
plt.show()

ClmProcedureCode = ['ClmProcedureCode_1', 'ClmProcedureCode_2', 'ClmProcedureCode_3',
        'ClmProcedureCode_4', 'ClmProcedureCode_5', 'ClmProcedureCode_6',]

ClmDiagnosisCode = ['ClmDiagnosisCode_1',
       'ClmDiagnosisCode_2', 'ClmDiagnosisCode_3', 'ClmDiagnosisCode_4',
       'ClmDiagnosisCode_5', 'ClmDiagnosisCode_6', 'ClmDiagnosisCode_7',
       'ClmDiagnosisCode_8', 'ClmDiagnosisCode_9', 'ClmDiagnosisCode_10']

# Fill NaN values for ClmDiagnosisCode columns in both DataFrames
inpatient_train[ClmDiagnosisCode] = inpatient_train[ClmDiagnosisCode].fillna('-1')
outpatient_train[ClmDiagnosisCode] = outpatient_train[ClmDiagnosisCode].fillna('-1')

# Function to count non '-1' diagnosis codes in each row
def count_diagnoses(df, code_list, index_name):
    # Apply a lambda function to count non '-1' codes row-wise
    df[index_name] = df[code_list].apply(lambda row: sum(code != '-1' for code in row), axis=1)

# Apply the counting function to both DataFrames
count_diagnoses(inpatient_train, ClmDiagnosisCode, 'DiagnosisCnt')
count_diagnoses(outpatient_train, ClmDiagnosisCode, 'DiagnosisCnt')

count_diagnoses(inpatient_train,ClmDiagnosisCode,'DiagnosisIndex')
count_diagnoses(outpatient_train,ClmDiagnosisCode,'DiagnosisIndex')
count_diagnoses(inpatient_train,ClmProcedureCode,'ProcedureIndex')
count_diagnoses(outpatient_train,ClmProcedureCode,'ProcedureIndex')

ClmDiagnosisCode.remove('ClmDiagnosisCode_1') 
inpatient_train.head()

count_diagnoses(inpatient_train,ClmDiagnosisCode,'DiagnosisIndex')
count_diagnoses(inpatient_train,ClmProcedureCode,'ProcedureIndex')
count_diagnoses(outpatient_train,ClmDiagnosisCode,'DiagnosisIndex')

inpatient_train.groupby('DiagnosisIndex').count()

def func(pct, allvalues):
    absolute = int(pct / 100. * np.sum(allvalues))
    return "{:.1f}% ({:d})".format(pct, absolute)

# Prepare data for pie chart
plt.figure(figsize=(8, 6), dpi=100)
data = inpatient_train.groupby('DiagnosisIndex').count().BeneID

# Create the pie chart
plt.pie(data,
        labels=data.index,
        autopct=lambda pct: func(pct, data.values))

# Set the title and display the plot
plt.title('DiagnosisIndex in Inpatient Train Data')
plt.show()


def func(pct, allvalues):
    absolute = int(pct / 100. * np.sum(allvalues))
    return "{:.1f}% ({:d})".format(pct, absolute)

# Prepare data for pie chart
plt.figure(figsize=(8, 6), dpi=100)
data = outpatient_train.groupby('DiagnosisIndex').count().BeneID

# Create the pie chart
plt.pie(data,
        labels=data.index,
        autopct=lambda pct: func(pct, data.values))

# Set the title and display the plot
plt.title('DiagnosisIndex in Outpatient Data')
plt.show()

inpatient_train.columns

def isSamePhysician(df):
    # Use vectorized comparison to create the 'SamePhysician' column
    df['SamePhysician'] = (df['AttendingPhysician'] == df['OperatingPhysician']).astype(int)

inpatient_train.OtherPhysician.fillna(0,inplace=True)
outpatient_train.OtherPhysician.fillna(0,inplace=True)

inpatient_test.OtherPhysician.fillna(0,inplace=True)
outpatient_test.OtherPhysician.fillna(0,inplace=True)

for df in [inpatient_train,outpatient_train,inpatient_test,outpatient_train]:
    for col in ['AttendingPhysician','OperatingPhysician']:
        df[col].dropna(inplace=True)

isSamePhysician(inpatient_train)
isSamePhysician(outpatient_train)

isSamePhysician(inpatient_test)
isSamePhysician(outpatient_test)

beneficiary_train.to_csv('bene_train.csv' ,sep='\t', encoding='utf-8')


# PROVIDER AND FRAUDS
label_train

provider_num = len(label_train['Provider'].unique())
provider_num

label_test

label_train['PotentialFraud'].replace('No',0,inplace=True)
label_train['PotentialFraud'].replace('Yes',1,inplace=True)

fraud = label_train[label_train['PotentialFraud']==1]
fraud.shape[0]

import matplotlib.pyplot as plt
import numpy as np

# Group by 'PotentialFraud' and count the number of providers
data = label_train.groupby('PotentialFraud').count()
labels = ['Prov. without potential fraud', 'Prov. with potential fraud']

# Plotting
plt.figure(figsize=(8, 6), dpi=100)
fig = plt.bar(x=data.index, height=data['Provider'], color=['navy', 'crimson'])
plt.title('Distribution Between Potential Fraud and Normal Provider')
plt.xticks(ticks=np.arange(len(labels)), labels=labels)

# Add percentage annotations on the bars
total_providers = label_train.shape[0]
for p in fig.patches:
    height = p.get_height()
    percentage = round((100 * height) / total_providers, 2)
    plt.annotate(f'{percentage}%', (p.get_x() + p.get_width() / 2, height), ha='center', va='bottom')

plt.show()

# List of columns to drop in each DataFrame
columns_to_drop_initial = ['ClaimStartDt', 'ClaimEndDt']
delete = ClmProcedureCode + ClmDiagnosisCode

# Drop specified columns from all DataFrames
for df in [inpatient_train, inpatient_test, outpatient_test, outpatient_train]:
    # Drop the initial set of columns if they exist
    if set(columns_to_drop_initial).issubset(df.columns):
        df.drop(columns=columns_to_drop_initial, inplace=True)
    
    # Drop the combined list of columns if they exist
    if set(delete).issubset(df.columns):
        df.drop(columns=delete, inplace=True)

common_cols = [col for col in outpatient_train.columns if col in inpatient_train.columns]
common_cols

inpatient_train["Admitted"] = 1
outpatient_train["Admitted"] = 0
inpatient_test["Admitted"] = 1
outpatient_test["Admitted"] = 0

# Check for non-numeric values
non_numeric_values = inpatient_train[~inpatient_train['DeductibleAmtPaid'].apply(lambda x: str(x).isdigit())]
print(non_numeric_values)

# Check for NaNs
nan_count = inpatient_train['DeductibleAmtPaid'].isna().sum()
print(f'Number of NaNs: {nan_count}')

# Replace non-numeric values with -9999
inpatient_train['DeductibleAmtPaid'] = pd.to_numeric(inpatient_train['DeductibleAmtPaid'], errors='coerce')

# Fill NaNs with -9999
inpatient_train['DeductibleAmtPaid'] = inpatient_train['DeductibleAmtPaid'].fillna(-9999)

# Convert to integer
inpatient_train['DeductibleAmtPaid'] = inpatient_train['DeductibleAmtPaid'].astype(int)

# Verify conversion
print(inpatient_train['DeductibleAmtPaid'].dtype)


ip_op_train = pd.merge(left=inpatient_train, right=outpatient_train, how='outer')
ip_op_test = pd.merge(left=inpatient_test, right=outpatient_test, how='outer')

ip_op_train = pd.merge(left=ip_op_train, right=label_train, on='Provider', how='inner')
ip_op_test = pd.merge(left=ip_op_test, right=label_test, on='Provider', how='inner')

# Joining the IP_OP dataset with the BENE data
train_df = pd.merge(left=ip_op_train, right=beneficiary_train, left_on='BeneID', right_on='BeneID',how='inner')
train_df.shape

test_df = pd.merge(left=ip_op_test, right=beneficiary_test, left_on='BeneID', right_on='BeneID',how='inner')
test_df.shape

outpatient_test.shape[0] +inpatient_test.BeneID.shape[0]
test_df.to_csv('test.csv' ,sep='\t', encoding='utf-8')
train_df.to_csv('train.csv' ,sep='\t', encoding='utf-8')

# MACHINE LEARNING
first_train = train_df.copy()
first_train.columns

obj_list = ['BeneID', 'ClaimID','Provider','AttendingPhysician', 'ClmAdmitDiagnosisCode','OperatingPhysician', 'OtherPhysician','DiagnosisGroupCode','ClmDiagnosisCode_1','SamePhysician']


# Initialize the LabelEncoder
labelencoder = LabelEncoder()

# Loop through each column in the DataFrame
for col in first_train.columns:
    # Check if the column has mixed types
    if first_train[col].apply(lambda x: isinstance(x, str)).any() and first_train[col].apply(lambda x: isinstance(x, (int, float))).any():
        # Convert all values to strings
        first_train[col] = first_train[col].astype(str)
    
    # Now apply the LabelEncoder
    first_train[col] = labelencoder.fit_transform(first_train[col])


X = first_train.drop(['PotentialFraud','DOB'],axis=1)
y = first_train['PotentialFraud']

X.info()
X = X.fillna(-9999)

X_train, X_val, y_train, y_val = tts(X, y, test_size=0.20, stratify=y, random_state=42)
# Checking shape of each set
X_train.shape, X_val.shape, y_train.shape, y_val.shape

# Checking count of tgt labels in y_train
y_train.value_counts()
X_val.head()

# RANDOM FOREST
# First Model
#random forest model creation 
rfc = RandomForestClassifier(n_estimators=500,class_weight='balanced',random_state=123,max_depth=4) 
rfc.fit(X_train, y_train) 
#predictions 
y_predict = rfc.predict(X_val)

fpr, tpr, thresholds = roc_curve(y_val, rfc.predict_proba(X_val)[:,1])
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=1, label='ROC curve (AUC = %0.2f)' % roc_auc)

for label in range(1,10,1):
    plt.text((10-label)/10,(10-label)/10,thresholds[label*15],fontdict={'size': 14})
    
plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic of First Model')
plt.legend(loc="lower right")
plt.show()

# Building Evaluation Parameters
n_outliers = len(fraud) 
n_errors = (y_predict != y_val).sum() 
print("The model used is Random Forest classifier") 
  
acc = accuracy_score(y_val, y_predict) 
print("The accuracy is {}".format(acc)) 
  
prec = precision_score(y_val, y_predict) 
print("The precision is {}".format(prec)) 
  
rec = recall_score(y_val,y_predict) 
print("The recall is {}".format(rec)) 
  
f1 = f1_score(y_val, y_predict) 
print("The F1-Score is {}".format(f1)) 
  
MCC = matthews_corrcoef(y_val, y_val) 
print("The Matthews correlation coefficient is{}".format(MCC)) 

# Second Model
#random forest model creation 
rfc2 = RandomForestClassifier(n_estimators=500,class_weight='balanced',random_state=123,max_depth=10) 
rfc2.fit(X_train, y_train) 
#predictions 
y_predict2 = rfc2.predict(X_val)

fpr, tpr, thresholds = roc_curve(y_val, rfc2.predict_proba(X_val)[:,1])
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=1, label='ROC curve (AUC = %0.2f)' % roc_auc)

for label in range(1,10,1):
    plt.text((10-label)/10,(10-label)/10,thresholds[label*15],fontdict={'size': 14})
    
plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic of Second Model')
plt.legend(loc="lower right")
plt.show()

print("The model used is Random Forest classifier with depth of 10") 
  
acc = accuracy_score(y_val, y_predict2) 
print("The accuracy is {}".format(acc)) 
  
prec = precision_score(y_val, y_predict2) 
print("The precision is {}".format(prec)) 
  
rec = recall_score(y_val,y_predict2) 
print("The recall is {}".format(rec)) 
  
f1 = f1_score(y_val, y_predict2) 
print("The F1-Score is {}".format(f1)) 

# Third Model
#random forest model creation 
rfc3 = RandomForestClassifier(n_estimators=500,class_weight='balanced',random_state=123,max_depth=15) 
rfc3.fit(X_train, y_train) 
#predictions 
y_predict3 = rfc3.predict(X_val)

fpr, tpr, thresholds = roc_curve(y_val, rfc3.predict_proba(X_val)[:,1])
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=1, label='ROC curve (AUC = %0.2f)' % roc_auc)

for label in range(1,10,1):
    plt.text((10-label)/10,(10-label)/10,thresholds[label*15],fontdict={'size': 14})
    
plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic of Third Model')
plt.legend(loc="lower right")
plt.show()

print("The model used is Random Forest classifier with depth of 15") 
  
acc = accuracy_score(y_val, y_predict3) 
print("The accuracy is {}".format(acc)) 
  
prec = precision_score(y_val, y_predict3) 
print("The precision is {}".format(prec)) 
  
rec = recall_score(y_val,y_predict3) 
print("The recall is {}".format(rec)) 
  
f1 = f1_score(y_val, y_predict3) 
print("The F1-Score is {}".format(f1)) 

# Fourth Model
#random forest model creation 
rfc4 = RandomForestClassifier(n_estimators=500,class_weight='balanced',random_state=123,max_depth=25) 
rfc4.fit(X_train, y_train) 
#predictions 
y_predict4 = rfc4.predict(X_val)

print("The model used is Random Forest classifier with depth of 25") 
  
acc = accuracy_score(y_val, y_predict4) 
print("The accuracy is {}".format(acc)) 
  
prec = precision_score(y_val, y_predict4) 
print("The precision is {}".format(prec)) 
  
rec = recall_score(y_val,y_predict4) 
print("The recall is {}".format(rec)) 
  
f1 = f1_score(y_val, y_predict4) 
print("The F1-Score is {}".format(f1))

fpr, tpr, thresholds = roc_curve(y_val, rfc4.predict_proba(X_val)[:,1])
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=1, label='ROC curve (AUC = %0.2f)' % roc_auc)

for label in range(1,10,1):
    plt.text((10-label)/10,(10-label)/10,thresholds[label*15],fontdict={'size': 14})
    
plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic of Fourth Model')
plt.legend(loc="lower right")
plt.show()

# Fifth Model
#random forest model creation 
rfc5 = RandomForestClassifier(n_estimators=1000,class_weight='balanced',random_state=123,max_depth=25) 
rfc5.fit(X_train, y_train) 
#predictions 
y_predict5 = rfc5.predict(X_val)


print("The model used is Random Forest classifier with depth of 25 and estimators of 1000") 
  
acc = accuracy_score(y_val, y_predict5) 
print("The accuracy is {}".format(acc)) 
  
prec = precision_score(y_val, y_predict5) 
print("The precision is {}".format(prec)) 
  
rec = recall_score(y_val,y_predict5) 
print("The recall is {}".format(rec)) 
  
f1 = f1_score(y_val, y_predict5) 
print("The F1-Score is {}".format(f1)) 

fpr, tpr, thresholds = roc_curve(y_val, rfc5.predict_proba(X_val)[:,1])
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=1, label='ROC curve (AUC = %0.2f)' % roc_auc)

for label in range(1,10,1):
    plt.text((10-label)/10,(10-label)/10,thresholds[label*15],fontdict={'size': 14})
    
plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic of Fifth Model')
plt.legend(loc="lower right")
plt.show()

# Sixth Model
#random forest model creation 
rfc6 = RandomForestClassifier(n_estimators=500,class_weight='balanced',random_state=123,max_depth=25,oob_score=True) 
rfc6.fit(X_train, y_train) 
#predictions 
y_predict6 = rfc6.predict(X_val)


print("The model used is Random Forest classifier with depth of 25 and estimators of 1000") 
  
acc = accuracy_score(y_val, y_predict6) 
print("The accuracy is {}".format(acc)) 
  
prec = precision_score(y_val, y_predict6) 
print("The precision is {}".format(prec)) 
  
rec = recall_score(y_val,y_predict6) 
print("The recall is {}".format(rec)) 
  
f1 = f1_score(y_val, y_predict6) 
print("The F1-Score is {}".format(f1)) 

cnt=0
for model in [rfc,rfc2,rfc4,rfc4,rfc5]:
    cnt+=1
    print("model",cnt)
    #print("used estimators:",model.estimators_[-1])
    print("feature_importances_",model.feature_importances_)
    print("num of feature seen during fitting :",model.n_features_in_,"\n")

rfc5.decision_path(X_val)

# SUPPORT VECTOR MACHINE
svm_model = svm.SVC()
svm_model.fit(X_train, y_train) 
#predictions 
y_predict_svm = svm_model.predict(X_val)

print("The model used is SVM (Support Vector Machines)") 
  
acc = accuracy_score(y_val, y_predict_svm) 
print("The accuracy is {}".format(acc)) 
  
prec = precision_score(y_val, y_predict_svm) 
print("The precision is {}".format(prec)) 
  
rec = recall_score(y_val,y_predict_svm) 
print("The recall is {}".format(rec)) 
  
f1 = f1_score(y_val, y_predict_svm) 
print("The F1-Score is {}".format(f1)) 

# K NEIGHBORS CLASSIFIER
knn_model = KNeighborsClassifier(n_neighbors = 3)
knn_model.fit(X_train, y_train) 
#predictions 
y_predict_knn = knn_model.predict(X_val)


print("The model used is SVM (Support Vector Machines)") 
  
acc = accuracy_score(y_val, y_predict_knn) 
print("The accuracy is {}".format(acc)) 
  
prec = precision_score(y_val, y_predict_knn) 
print("The precision is {}".format(prec)) 
  
rec = recall_score(y_val,y_predict_knn) 
print("The recall is {}".format(rec)) 
  
f1 = f1_score(y_val, y_predict_knn) 
print("The F1-Score is {}".format(f1)) 