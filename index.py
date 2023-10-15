import pickle
import joblib
import numpy as np
from flask import Flask, render_template, request
import pandas as pd
df = pd.read_csv('modified_ipl_data.csv')
columnsname=['venue','bowl_team','bat_team']
uniq_v=  list(df['venue'].unique())
uniq_bo= list(df['bowl_team'].unique())
uniq_ba= list(df['bat_team'].unique())

regressor = joblib.load('iplmodel_ridge.sav')
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

app = Flask(__name__)

@app.route('/')
def homepage():
    return render_template('index.html')

@app.route('/home')
def home():
        return render_template('home.html', batting_team=uniq_ba, bowling_team=uniq_bo, venues=uniq_v)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/teams')
def teams():
    return render_template('team.html')


@app.route('/predict', methods=['POST'])
def predict():
    a = []

    if request.method == 'POST':


        venue = request.form['venue']
        print("Venue", venue)
        if venue == 'M Chinnaswamy Stadium, Bengaluru':
           a = a + [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        elif venue == 'Punjab Cricket Association Stadium, Mohali':
           a = a + [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        elif venue == 'Feroz Shah Kotla':
           a = a + [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        elif venue == 'Wankhede Stadium, Mumbai':
           a = a + [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        elif venue == 'Eden Gardens, Kolkata':
           a = a + [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        elif venue == 'Sawai Mansingh Stadium, Jaipur':
           a = a + [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        elif venue == 'Rajiv Gandhi International Stadium, Uppal, Hyderabad':
           a = a + [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        elif venue == 'MA Chidambaram Stadium, Chepauk, Chennai':
           a = a + [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        elif venue == 'Dr DY Patil Sports Academy, Mumbai':
           a = a + [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        elif venue == 'Newlands':
           a = a + [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        elif venue == "St George's Park":
           a = a + [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        elif venue == 'Kingsmead':
           a = a + [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        elif venue == 'SuperSport Park':
           a = a + [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        elif venue == 'Buffalo Park':
           a = a + [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        elif venue == 'New Wanderers Stadium':
           a = a + [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        elif venue == 'De Beers Diamond Oval':
           a = a + [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        elif venue == 'OUTsurance Oval':
           a = a + [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        elif venue == 'Brabourne Stadium, Mumbai':
           a = a + [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        elif venue == 'Sardar Patel Stadium, Motera':
           a = a + [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        elif venue == 'Barabati Stadium':
           a = a + [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        elif venue == 'Vidarbha Cricket Association Stadium, Jamtha':
           a = a + [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        elif venue == 'Himachal Pradesh Cricket Association Stadium, Dharamsala':
           a = a + [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        elif venue == 'Nehru Stadium':
           a = a + [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        elif venue == 'Holkar Cricket Stadium':
           a = a + [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        elif venue == 'Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium':
           a = a + [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        elif venue == 'Subrata Roy Sahara Stadium':
           a = a + [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        elif venue == 'Maharashtra Cricket Association Stadium, Pune':
           a = a + [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        elif venue == 'Shaheed Veer Narayan Singh International Stadium':
           a = a + [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        elif venue == 'JSCA International Stadium Complex':
           a = a + [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        elif venue == 'Sheikh Zayed Stadium':
           a = a + [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        elif venue == 'Sharjah Cricket Stadium':
           a = a + [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
        elif venue == 'Dubai International Cricket Stadium':
           a = a + [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
        elif venue == 'Saurashtra Cricket Association Stadium':
           a = a + [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
        elif venue == 'Green Park':
           a = a + [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
        elif venue == 'Arun Jaitley Stadium, Delhi':
           a = a + [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
        elif venue == 'Narendra Modi Stadium, Ahmedabad':
           a = a + [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
        elif venue == 'Zayed Cricket Stadium, Abu Dhabi':
           a = a + [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
        elif venue == 'Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium, Lucknow':
           a = a + [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
        elif venue == 'Barsapara Cricket Stadium, Guwahati':
           a = a + [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]


        batting_team = request.form['batting-team']
        if batting_team == 'Chennai Super Kings':
            a = a + [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif batting_team == 'Delhi Capitals':
            a = a + [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif batting_team == 'Kings XI Punjab':
            a = a + [0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif batting_team == 'Kolkata Knight Riders':
            a = a + [0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif batting_team == 'Mumbai Indians':
            a = a + [0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0]
        elif batting_team == 'Rajasthan Royals':
            a = a + [0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0]
        elif batting_team == 'Royal Challengers Bangalore':
            a = a + [0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0]
        elif batting_team == 'Sunrisers Hyderabad':
            a = a + [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0]
        elif batting_team == 'Delhi Daredevils':
            a = a + [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0]
        elif batting_team == 'Deccan Chargers':
            a=  a + [0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0]
        elif batting_team == 'Kochi Tuskers Kerela':
            a=  a + [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0]
        elif batting_team == 'Pune Warriors':
            a = a + [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0]
        elif batting_team == 'Rising Pune Supergiant':
            a = a + [0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0]
        elif batting_team == 'Gujarat Lions':
            a=  a + [0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0]
        elif batting_team == 'Punjab Kings':
            a=  a + [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0]
        elif batting_team == 'Gujarat Titans':
            a=  a + [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0]
        elif batting_team == 'Lucknow Super Giants':
            a=  a + [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]

        bowling_team = request.form['bowling-team']
        if bowling_team == 'Chennai Super Kings':
            a = a + [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        elif bowling_team == 'Delhi Capitals':
            a = a + [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        elif bowling_team == 'Kings XI Punjab':
            a = a + [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        elif bowling_team == 'Kolkata Knight Riders':
            a = a + [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        elif bowling_team == 'Mumbai Indians':
            a = a + [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        elif bowling_team == 'Rajasthan Royals':
            a = a + [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        elif bowling_team == 'Royal Challengers Bangalore':
            a = a + [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        elif bowling_team == 'Sunrisers Hyderabad':
            a = a + [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        elif bowling_team == 'Delhi Daredevils':
            a = a + [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
        elif bowling_team == 'Deccan Chargers':
            a = a + [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
        elif bowling_team == 'Kochi Tuskers Kerela':
            a = a + [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
        elif bowling_team == 'Pune Warriors':
            a = a + [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
        elif bowling_team == 'Rising Pune Supergiant':
            a = a + [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
        elif bowling_team == 'Gujarat Lions':
            a = a + [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
        elif bowling_team == 'Punjab Kings':
            a = a + [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
        elif bowling_team == 'Gujarat Titans':
            a = a + [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
        elif bowling_team == 'Lucknow Super Giants':
            a = a + [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]

        if batting_team == bowling_team and batting_team != 'none' and bowling_team != 'none':
            return render_template('home.html',
                                   val='Batting team and Bowling team cant be same and none of the values can\'t be empty.')

        over = request.form['overs']
        runs = request.form['runs']
        wicketi = request.form['wickets']
        wicki = request.form['wickets_in_prev_5']
        running = request.form['runs_in_prev_5']

        print("Over",over,"Runi",runs,"running",running,"wicki",wicki)

        if over == '' or runs == '' or wicketi == '' or wicki == '' or running == '':
            return render_template('home.html', val='You can\'t leave any field empty!!!')

        overs = float(over)
        runs = int(runs)
        wickets = int(wicketi)
        totalruns_last_5 = int(running)
        wickets_last_5 = int(wicki)

        a = np.array(a).reshape(1, -1)

        b = [runs, wickets, overs, totalruns_last_5, wickets_last_5]
        print("B=",b)
        b = np.array(b).reshape(1, -1)
        b = scaler.transform(b)

        data = np.concatenate((a, b), axis=1)
        print(data[0])


        my_prediction = int(regressor.predict(data)[0])
        print("my prediction",my_prediction)

        return render_template('home.html',
                               val=f'The final score will be around {my_prediction - 5} to {my_prediction + 10}.')


if __name__ == '__main__':
    app.run(debug=True)
