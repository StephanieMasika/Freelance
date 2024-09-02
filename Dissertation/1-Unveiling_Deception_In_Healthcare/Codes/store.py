# line 145 main code - model3
for df in [df_train, df_test]:
    for i in range(df.shape[0]):#
        chronicDCnt = 0
        for disease in diseases:
            if df_train.loc[i,disease] > 1:
                 chronicDCnt+=1
            else:
                 continue
        df.loc[i,'ChronicDiseaseIndex'] = chronicDCnt


# line 178
for df in [inpatient_train,inpatient_test,outpatient_test,outpatient_train]:  
        for col in ['ClaimStartDt','ClaimEndDt']:
            df[col] = pd.to_datetime(df[col])
            
        df['ClaimPeriod'] =  round(((df.ClaimEndDt - df.ClaimStartDt).dt.days),0)
            
for df in [inpatient_train,inpatient_test]:
            df.AdmissionDt = pd.to_datetime(df.AdmissionDt)
            df.DischargeDt = pd.to_datetime(df.DischargeDt)
            df['TimeInHptal'] =  round(((df.DischargeDt - df.AdmissionDt).dt.days),0)
            df.drop(['DischargeDt','AdmissionDt'],axis=1,inplace=True)

# line 217
for code in ClmDiagnosisCode:
    inpatient_train.fillna('-1',inplace=True)
    outpatient_train.fillna('-1',inplace=True)

def CountingIndex(df,codeList,indexName):
    
    for i in range(df.shape[0]):#
        DiagnosisCnt = 0
        for code in codeList:
            if df.loc[i,code]== '-1':
               
                continue
                 
            else:
                 DiagnosisCnt+=1
        df.loc[i,indexName] = DiagnosisCnt

# line 244
plt.figure(figsize=(8,6),dpi=100)
data = inpatient_train.groupby('DiagnosisIndex').count().BeneID

fig = plt.pie(data,
              labels=data.index,
              autopct = lambda pct: func(pct, data.values),)
plt.title('DiagnosisIndex in outpatient data')

plt.title('DiagnosisIndex in Inpatient Train Data')
plt.show()

# line 262
plt.figure(figsize=(8,6),dpi=100)
data = outpatient_train.groupby('DiagnosisIndex').count().BeneID

fig = plt.pie(data,
              labels=data.index,
              autopct = lambda pct: func(pct, data.values),)
plt.title('DiagnosisIndex in outpatient data')

plt.show()

# line 281
def isSamePhysician(df):
    for i in range(0,df.shape[0]):

        if  df.loc[i,['AttendingPhysician']][0] == df.loc[i,['OperatingPhysician']][0]:
            df.loc[i,'SamePhysician'] = 1
        else:
            df.loc[i,'SamePhysician'] = 0

# line 318
label_train.groupby('PotentialFraud').count()
plt.figure(figsize=(8,6),dpi=100)
data=label_train.groupby('PotentialFraud').count()
labels=['Prov. without potential fraud','Prov. with potential fraud']
fig = plt.bar(x=data.index, height = data.Provider,color=['navy','crimson'])
plt.title('the dsitribution between potential fraud and normal provider')
x_pos = np.arange(2)
plt.xticks(x_pos, labels=labels)

cnt = 0
for p in fig.patches:
    height = p.get_height()
    width = p.get_width()
    x, y = p.get_xy()
    #plt.annotate(label[cnt],(x+width/2,y+height),ha='center')
    plt.annotate(f'{str(round((100*height)/(label_train.shape[0]),2))+"%"}',(x+width/2,y+height),ha='center')
    cnt+=1

plt.show()

# line 340
for df in [inpatient_train,inpatient_test,outpatient_test,outpatient_train]:  
    try:
        df.drop(['ClaimStartDt','ClaimEndDt'],axis=1,inplace=True)        
    except:
        continue

delete = ClmProcedureCode+ClmDiagnosisCode
for df in [inpatient_train,inpatient_test,outpatient_test,outpatient_train]: 
    try:
        df.drop(delete,axis=1,inplace=True)
    except:
        continue


# line 407
labelencoder = LabelEncoder()
for col in obj_list:
    first_train[col] = labelencoder.fit_transform(first_train[col])