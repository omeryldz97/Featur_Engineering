
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
#!pip install missingno
import missingno as msno
from datetime import date
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier

pd.set_option("display.max_columns",None)
pd.set_option("display.max_rows",None)
pd.set_option("display.float_format", lambda x:"%.3f" %x)
pd.set_option("display.width",500)

df=pd.read_csv("") #The dataset cannot be shared because it is private
df.head()


def check_df(dataframe,head=5):
    print("##################### Shape ###################")
    print(dataframe.shape)
    print("######################## head ##################")
    print(dataframe.head(head))
    print("##################### dtypes ###################")
    print(dataframe.dtypes)
    print("####################### tail ###################")
    print(dataframe.tail(head))
    print("###################### NA ####################")
    print(dataframe.isnull().sum())
    print("######################## Quantiles ############")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

check_df(df)



def cat_summary(dataframe,col_name,plot=False):
    print(pd.DataFrame({col_name:dataframe[col_name].value_counts(), "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################")
    if plot:
        sns.countplot(x=dataframe[col_name],data=dataframe)
        plt.show()
cat_summary(df,"Outcome")


def num_summary(dataframe,numerical_col,plot=False):
    quantiles= [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show()

for i in num_cols:
    num_summary(df,i,plot=True)


def target_summary_with_sum(dataframe,target,num_cols):
    print(dataframe.groupby(target).agg({num_cols:"mean"}),end="\n\n\n")

for i in num_cols:
    target_summary_with_sum(df,"Outcome",i)


df.corr()

f, ax = plt.subplots(figsize=[18, 13])
sns.heatmap(df.corr(), annot=True, fmt=".2f", ax=ax, cmap="magma")
ax.set_title("Correlation Matrix", fontsize=20)
plt.show(block=True)

# Feature Engineering

zero_columns=[i for i in df.columns if (df[i].min()==0 and i not in ["Pregnancies", "Outcome"])]

for i in zero_columns:
    df[i]=np.where(df[i]==0,np.nan,df[i])


for i in zero_columns:
    df.loc[df[i].isnull(),i]=df[i].median()

df.isnull().sum()

df.loc[(df["Age"]>=21) & df["Age"]<50,"New_Age_Cat"]="mature"
df.loc[(df["Age"]>=50) ,"New_Age_Cat"]="senior"

df["New_BMI"]=pd.cut(x=df["BMI"],bins=[0,18.5,24.9,29.9,100],labels=["Underweight","Healthy","Overweight","Obese"])

df["New_Glucose"]=pd.cut(x=df["Glucose"],bins=[0,140,200,300],labels=["Normal","Prediabetes", "Diabetes"])

df.loc[(df["Age"]>=21) & (df["BMI"]<18.5) & (df["Age"]<50),"New_Age_Bmı_Nom"]="underweightmature"
df.loc[(df["Age"]>50) & (df["BMI"]<18.5), "New_Age_Bmı_Nom"]="underweightsenior"
df.loc[(df["Age"]>21) & (df["BMI"]>=18.5) & (df["BMI"]<25) ,"New_Age_Bmı_Nom"]="healthymature"
df.loc[(df["Age"]>=50) & (df["BMI"]>=18.5) & (df["BMI"]<25),"New_Age_Bmı_Nom"]="healthysenior"
df.loc[(df["Age"]>=21) & (df["Age"]<50) & (df["BMI"]>=25) & (df["BMI"]<30),"New_Age_Bmı_Nom"]="overweightymature"
df.loc[(df["Age"]>=50) & (df["BMI"]>=25) & (df["BMI"]<30),"New_Age_Bmı_Nom"]="overweightysenior"
df.loc[(df["Age"]>=21) & (df["Age"]<50) & (df["BMI"]>18.5),"New_Age_Bmı_Nom"]="obesemature"
df.loc[(df["BMI"] > 18.5) & (df["Age"] >= 50), "New_Age_Bmı_Nom"] = "obesesenior"

df.loc[(df["Glucose"]<70) & ((df["Age"]>=21) & (df["Age"]<50)),"New_Age_Glucose_Nom"]="lowmature"
df.loc[(df["Age"]>=50) & (df["Glucose"]<70),"New_Age_Glucose_Nom"]="lowsenior"
df.loc[((df["Age"]>=21) & (df["Age"]<50)) & ((df["Glucose"]>=70) & (df["Glucose"]<100)),"New_Age_Glucose_Nom"]="normalmature"
df.loc[((df["Glucose"]>=70) & (df["Glucose"]<100)) & (df["Age"]>=50),"New_Age_Glucose_Nom"]="normalsenior"
df.loc[((df["Glucose"]>=100) & (df["Glucose"]<125)) & ((df["Age"]>=21) & (df["Age"]<50)),"New_Age_Glucose_Nom"]="hiddenmature"
df.loc[((df["Glucose"]>=100) & (df["Glucose"]<=125)) & (df["Age"]>=50),"New_Age_Glucose_Nom"]="hiddensenior"
df.loc[(df["Glucose"]>125) & ((df["Age"]>=21) & (df["Age"]<50)),"New_Age_Glucose_Nom"]="highmature"
df.loc[(df["Glucose"]>125) & (df["Age"]>=50),"New_Age_Glucose_Nom"]="highsenior"


def set_insulin(dataframe,col_name="Insulin"):
    if 16<= dataframe[col_name]<=166:
        return "Normal"
    else:
        return "Abnormal"

df["New_Insulin_Score"]=df.apply(set_insulin,axis=1)
df["New_Glucose*Insulin"]=df["Glucose"] * df["Insulin"]

df.columns=[i.upper() for i in df.columns]
df.head()


def label_encoder(dataframe,binary_col):
    Labelencoder=LabelEncoder()
    dataframe[binary_col]=Labelencoder.fit_transform(dataframe[binary_col])
    return dataframe
binary_col=[i for i in df.columns if df[i].dtypes=="O" and df[i].nunique()==2]

for i in binary_col:
    df=label_encoder(df,i)



def grab_col_names(dataframe, cat_th=10, car_th=20):
    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols, num_cols, cat_but_car
cat_cols, num_cols, cat_but_car = grab_col_names(df)

cat_cols=[i for i in cat_cols if i not in binary_col and i not in ["OUTCOME"]]

def one_hot_encoder(dataframe,categorical_cols,drop_first=False):
    dataframe=pd.get_dummies(dataframe,columns=categorical_cols,drop_first=drop_first)
    return dataframe
df=one_hot_encoder(df,cat_cols,drop_first=True)
df.head()

num_cols

scaler=StandardScaler()
df[num_cols]=scaler.fit_transform(df[num_cols])
df.head()
df.shape


y=df["OUTCOME"]
X=df.drop("OUTCOME",axis=1)
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.30,random_state=17)
rf_model=RandomForestClassifier(random_state=16).fit(X_train,y_train)
y_pred=rf_model.predict(X_test)
y_pred

print(f"Accuracy: {round(accuracy_score(y_pred, y_test), 2)}")
print(f"Recall: {round(recall_score(y_pred,y_test),3)}")
print(f"Precision: {round(precision_score(y_pred,y_test), 2)}")
print(f"F1: {round(f1_score(y_pred,y_test), 2)}")
print(f"Auc: {round(roc_auc_score(y_pred,y_test), 2)}")

#Accuracy: 0.8
#Recall: 0.736
#Precision: 0.65
#F1: 0.69
#Auc: 0.78

#FEATURE IMPORTANCE

def plot_importance(model,features,num=len(X),save=False):
    feature_imp=pd.DataFrame({"Value":model.feature_importances_,"Feature":features.columns})
    print(feature_imp.sort_values("Value",ascending=False))
    plt.figure(figsize=(10,10))
    sns.set(font_scale=1)
    sns.barplot(x="Value",y="Feature",data=feature_imp.sort_values(by="Value",ascending=False)[0:num])

    plt.title("Features")
    plt.tight_layout()
    plt.show(block=True)
    if save:
        plt.savefig("importance.png")
plot_importance(rf_model,X)
