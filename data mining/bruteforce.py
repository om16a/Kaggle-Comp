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

data = pd.read_csv("newtrain.csv")
arr = ["id","Gender","Age","Driving_License","Region_Code","Previously_Insured","Vehicle_Age","Vehicle_Damage","Annual_Premium","Policy_Sales_Channel","Vintage","Response"]
n = 12
mx = 0.0
mxnum = 0
arrtmp= []
for i in range(1<<n -1):
    if(i == 0):
        i = 1
    arr2 = []
    for j in range(12):
        if((1<<j)&i):
            arr2.append(arr[j])

    # Select features and target
    col = data[arr2]
    res = data["Response"]
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(col, res, test_size=0.2, random_state=42)
    
    # Train the Decision Tree model
    classifier = XGBClassifier(n_estimators=300, random_state=10, scale_pos_weight=5, use_label_encoder=False, eval_metric="auc")
    classifier.fit(X_train, y_train)

    # Predict on the test set
    predictions = classifier.predict(X_test)
    # Calculate accuracy
    accuracy = accuracy_score(y_test, predictions)
    if(accuracy > mx):
        mxnum = i
        mx = accuracy
        # Save predictions for the test split to a file
        # output = X_test.copy()
        # output["Result"] = predictions
        # output.to_csv("test_split_predictions.csv", index=False)


for j in range(12):
        if((1<<j)&mxnum):
            arrtmp.append(arr[j])

    # Select features and target
col = data[arrtmp]
res = data["Response"]
classifier = XGBClassifier(n_estimators=300, random_state=10, scale_pos_weight=5, use_label_encoder=False, eval_metric="auc")
classifier.fit(col, res)

print(f"Accuracy: {mx:.2f}")
print(f"Max bimas : {mxnum}")
#lead the test file 
testdata = pd.read_csv("test.csv")
tstcol = testdata[arrtmp]
# tstcolsc = scaler.fit_transform(tstcol)
print(arrtmp)
#predict 
# predict = model.predict(tstcolsc)
predict = classifier.predict(tstcol)


#save file 
output = testdata[["id"]].copy() 
output["Result"] = predict    
output.to_csv("output.csv", index=False) 

print("Predictions saved to output.csv")


print("Predictions saved to test_split_predictions.csv")



