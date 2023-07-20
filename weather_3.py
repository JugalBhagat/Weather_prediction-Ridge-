import pandas as pd
import matplotlib as mpl
import sklearn
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
# %%
weather =pd.read_csv("weather.csv",index_col="DATE")

#------------------------------------------------------------- data cleaning---------------------------------------------------

#print("total fields with null values : ",weather.apply(pd.isnull).sum())
#print("total no of rows : ",weather.shape[0])

#----------------------- finding null percenatge in database------------------------------

null_per=weather.apply(pd.isnull).sum()/weather.shape[0]
#print("null percentage of data:",null_per)
valid_colm=weather.columns[null_per < 0.5]
#print(type(valid_colm))

#------------------------------updating database---------------------------------------------

weather=weather[valid_colm].copy()
weather.columns=weather.columns.str.lower()
#print(weather)
#weather=weather.ffill()    #it will field last value in ther null values
#print(weather.dtypes)      # data type of weather fields

weather.index=pd.to_datetime(weather.index)  # converting DATE (object -> Date type)

weather.index.year.value_counts().sort_index()  #sorting year values (so that we can count how many data we have for each year)
#print(weather.index.year.value_counts().sort_index())


#print(weather["snwd"].plot())

#---------add new column ["target"]------------------

weather["target_tmax"]=weather.shift(-1)["tmax"]

weather["target_tmin"]=weather.shift(-1)["tmin"]


weather=weather.ffill()


#print(weather)
rr=Ridge(alpha=.1)
#print(rr)
weather=weather.drop("wsf2",axis=1)
weather=weather.drop("pgtm",axis=1)
weather=weather.drop("fmtm",axis=1)
weather=weather.drop("awnd",axis=1)
weather=weather.drop("wdf2",axis=1)

predictors=weather.columns[~weather.columns.isin(["target_tmax","target_tmin","name","station"])]

#print(predictors)
#print(weather)

def my_backtest_tmax(weather,model,predictors,start=3650,step=90):  #skipping first 10 years
    all_p=[]  #list of dataframes of 90 days (predictions)
    for i in range(start,weather.shape[0],step):

        train=weather.iloc[:i,:] # till (I) [past data]
        test=weather.iloc[i:(i+step),:] # after (I) [future data]

        model.fit(train[predictors],train["target_tmax"]) #calling function (giving predictors,and set target)

        preds=model.predict(test[predictors]) # it will predict
        preds=pd.Series(preds,index=test.index) # converting to panda series cuz it is easy work with
        combine=pd.concat([test["target_tmax"],preds],axis=1)   # combine actual data and predictions and making it saperate columns (axis=1)
        combine.columns=["actual_tmax","prediction_tmax"] # setting column names
        combine["diff_tmax"]=(combine["prediction_tmax"]-combine["actual_tmax"]).abs() # finding diffrence
    
        all_p.append(combine)
    return pd.concat(all_p)

def my_backtest_tmin(weather,model,predictors,start=3650,step=90):  #skipping first 10 years
    all_p=[]  #list of dataframes of 90 days (predictions)
    for i in range(start,weather.shape[0],step):

        train=weather.iloc[:i,:] # till (I) [past data]
        test=weather.iloc[i:(i+step),:] # after (I) [future data]

        model.fit(train[predictors],train["target_tmin"]) #calling function (giving predictors,and set target)

        preds=model.predict(test[predictors]) # it will predict
        preds=pd.Series(preds,index=test.index) # converting to panda series cuz it is easy work with
        combine=pd.concat([test["target_tmin"],preds],axis=1)   # combine actual data and predictions and making it saperate columns (axis=1)
        combine.columns=["actual_tmin","prediction_tmin"] # setting column names
        combine["diff_tmin"]=(combine["prediction_tmin"]-combine["actual_tmin"]).abs() # finding diffrence
    
        all_p.append(combine)
    return pd.concat(all_p)

#my_predict_tmax=my_backtest_tmax(weather,rr,predictors)
#print(my_predict)
#print(my_predict["diff"].mean())

def pct_diff(old,new):
    return (new-old)/old

def compute_rolling(weather,horizon,col):
    label=f"rolling_{horizon}_{col}"
    weather[label]=weather[col].rolling(horizon).mean()
    weather[f"{label}_pct"]=pct_diff(weather[label],weather[col])
    return weather

rolling_horizon=[1,5,10,15]

for horizon in rolling_horizon:
    for col in ["tmax","tmin","prcp"]:
        weather=compute_rolling(weather,horizon,col)

weather=weather.iloc[15:,:]
weather=weather.fillna(0)

def expand_mean(df):
    return df.expanding(1).mean()

for col in ["tmax","tmin","prcp"]: 
    weather[f"month_avg_{col}"]=weather[col].groupby(weather.index.month,group_keys=False).apply(expand_mean)
    weather[f"day_avg_{col}"]=weather[col].groupby(weather.index.day_of_year,group_keys=False).apply(expand_mean)
    

#print(weather)

predictors=weather.columns[~weather.columns.isin(["target_tmax","target_tmin","name","station"])]

#print(predictors)

my_predict_tmax=my_backtest_tmax(weather,rr,predictors)
my_predict_tmin=my_backtest_tmin(weather,rr,predictors)


#df = weather.loc[:, ["DATE","station","name","prcp","snow","snwd","tmax","target_tmax","tmin","target_tmin"]]
df = weather.iloc[:, :9]

print(df)
#print(my_predict_tmax.sort_values("diff_tmax",ascending=False))
#print(my_predict_tmin.sort_values("diff_tmin",ascending=False))

print("max temp diff : ",my_predict_tmax["diff_tmax"].mean())
print("min temp diff : ",my_predict_tmin["diff_tmin"].mean())


print(my_predict_tmax["diff_tmax"].round().value_counts().plot())
print(my_predict_tmin["diff_tmin"].round().value_counts().plot())


# %%
