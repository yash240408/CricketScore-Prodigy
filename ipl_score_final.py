import joblib
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error

df = pd.read_csv('modified_ipl_data.csv')
df.head(2)
cols_to_drop = ['mid','batsman','bowler','innings']
df.drop(cols_to_drop,axis=1,inplace=True)

# we don't want first five overs data
df = df[df['overs']>=5.0]


df_new = pd.get_dummies(data=df,columns=['venue','bat_team','bowl_team'])


df_new = df_new[['date','venue_M Chinnaswamy Stadium, Bengaluru','venue_Punjab Cricket Association Stadium, Mohali',
                 'venue_Nehru Stadium','venue_Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium',
       'venue_Barabati Stadium', 'venue_Dr DY Patil Sports Academy, Mumbai',
       'venue_Brabourne Stadium, Mumbai','venue_Narendra Modi Stadium, Ahmedabad','venue_Vidarbha Cricket Association Stadium, Jamtha',
       'venue_Dubai International Cricket Stadium','venue_Barsapara Cricket Stadium, Guwahati',
       'venue_Eden Gardens, Kolkata', 'venue_Feroz Shah Kotla','venue_OUTsurance Oval',
       'venue_Himachal Pradesh Cricket Association Stadium, Dharamsala',
       'venue_Holkar Cricket Stadium','venue_Subrata Roy Sahara Stadium',
       'venue_JSCA International Stadium Complex','venue_New Wanderers Stadium',
       'venue_Green Park',
       'venue_MA Chidambaram Stadium, Chepauk, Chennai','venue_Buffalo Park',
       'venue_Maharashtra Cricket Association Stadium, Pune','venue_De Beers Diamond Oval',
       'venue_SuperSport Park',
       'venue_Rajiv Gandhi International Stadium, Uppal, Hyderabad',
       'venue_Sardar Patel Stadium, Motera','venue_Shaheed Veer Narayan Singh International Stadium',
       'venue_Sawai Mansingh Stadium, Jaipur','venue_Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium, Lucknow',
       'venue_Sharjah Cricket Stadium','venue_Arun Jaitley Stadium, Delhi',"venue_Saurashtra Cricket Association Stadium",
       'venue_Sheikh Zayed Stadium','venue_Zayed Cricket Stadium, Abu Dhabi',
       'venue_Wankhede Stadium, Mumbai','venue_Newlands',"venue_St George's Park",'venue_Kingsmead',

       'bat_team_Chennai Super Kings','bat_team_Rising Pune Supergiant','bat_team_Kochi Tuskers Kerala',
       'bat_team_Delhi Daredevils', 'bat_team_Deccan Chargers','bat_team_Punjab Kings','bat_team_Gujarat Titans',
       'bat_team_Kolkata Knight Riders', 'bat_team_Mumbai Indians','bat_team_Delhi Capitals',
       'bat_team_Rajasthan Royals', 'bat_team_Royal Challengers Bangalore','bat_team_Lucknow Super Giants',
       'bat_team_Sunrisers Hyderabad','bat_team_Pune Warriors','bat_team_Gujarat Lions', 'bat_team_Kings XI Punjab',

       'bowl_team_Chennai Super Kings','bowl_team_Deccan Chargers','bowl_team_Pune Warriors','bowl_team_Lucknow Super Giants',
       'bowl_team_Delhi Daredevils', 'bowl_team_Kings XI Punjab','bowl_team_Kochi Tuskers Kerala',
       'bowl_team_Kolkata Knight Riders', 'bowl_team_Mumbai Indians','bowl_team_Rising Pune Supergiant',
       'bowl_team_Rajasthan Royals', 'bowl_team_Royal Challengers Bangalore','bowl_team_Gujarat Titans',
       'bowl_team_Sunrisers Hyderabad','bowl_team_Gujarat Lions','bowl_team_Delhi Capitals','bowl_team_Punjab Kings',

       'overs','total','wickets_last_5','runs','wickets_sum','runs_last_5']]

df_new.reset_index(inplace=True)
df_new.drop('index',inplace=True,axis=1)

df_new.head(2)

scaler = StandardScaler()
scaled_cols = scaler.fit_transform(df_new[['runs', 'wickets_sum', 'overs', 'runs_last_5', 'wickets_last_5']])
pickle.dump(scaler, open('scaler.pkl','wb'))

scaled_cols = pd.DataFrame(scaled_cols,columns=['runs', 'wickets_sum', 'overs', 'runs_last_5', 'wickets_last_5'])
df_new.drop(['runs', 'wickets_sum', 'overs', 'runs_last_5', 'wickets_last_5'],axis=1,inplace=True)
df_new = pd.concat([df_new,scaled_cols],axis=1)


X_train = df_new.drop('total',axis=1)[df_new['date']<=2020]
X_test = df_new.drop('total',axis=1)[df_new['date']>=2021]

X_train.drop('date',inplace=True,axis=1)
X_test.drop('date',inplace=True,axis=1)


y_train = df_new[df_new['date']<=2020]['total'].values
y_test = df_new[df_new['date']>=2021]['total'].values

ridge = Ridge()
parameters={'alpha':[1e-3,1e-2,1,5,10,20]}
ridge_regressor = RandomizedSearchCV(ridge,parameters,cv=10,scoring='neg_mean_squared_error')
ridge_regressor.fit(X_train,y_train)

print(ridge_regressor.best_params_)
print(ridge_regressor.best_score_)

prediction_r = ridge_regressor.predict(X_test)
print('MAE:', mean_absolute_error(y_test, prediction_r))
print('MSE:', mean_squared_error(y_test, prediction_r))
print('RMSE:', np.sqrt(mean_squared_error(y_test, prediction_r)))
print(f'r2 score of ridge : {r2_score(y_test,prediction_r)}')

joblib.dump(ridge_regressor,'iplmodel_ridge.sav')