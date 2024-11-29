import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import  LabelEncoder
from xgboost import XGBClassifier 
scaler = StandardScaler()
#       0      1       2          3              4                   5               6              7                8                   9                 10         11                 
arr = ["id","Gender","Age","Driving_License","Region_Code","Previously_Insured","Vehicle_Age","Vehicle_Damage","Annual_Premium","Policy_Sales_Channel","Vintage","Response"]
# mas = 676|2 |(1<<2) | (1<<4) | (1<<5) | (1<<10)
mas = ((1<<11)-2)#^(1<<9)
# mas = (1<<3)
arr2= []
for i in range(12):
    if((1<<i)&mas):
        arr2.append(arr[i])
    
#load the data 
data = pd.read_csv("newtrain.csv");
data.dropna(inplace=True) # drop the null
col = data[arr2]
res = data["Response"]
# colsc = scaler.fit_transform(col)


# scaler = StandardScaler()
# col = pd.DataFrame(scaler.fit_transform(col), columns=arr2)    
#train the program 

# model = RandomForestClassifier(n_estimators=1000, random_state=10, class_weight="balanced")
model = XGBClassifier(n_estimators=300, random_state=10, scale_pos_weight=5, use_label_encoder=False, eval_metric="auc")
model.fit(col, res)
# # classifier = DecisionTreeClassifier(criterion='entropy', max_depth=120, random_state=42)
# classifier.fit(col , res)

testdata = pd.read_csv("newtest.csv")
tstcol = testdata[arr2]
# tstcolsc = scaler.fit_transform(tstcol)
print(arr2)
#predict 
# predict = model.predict(tstcolsc)
# predict = model.predict(tstcol)

# tstcol = pd.DataFrame(scaler.transform(tstcol), columns=arr2)
#save file 
output = testdata[["id"]].copy() 
output["Result"] = model.predict(tstcol)    
output.to_csv("output.csv", index=False) 

print("Predictions saved to output.csv")
