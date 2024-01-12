import pandas as pd
import numpy as np
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
import random

#NUMBER_KFOLDS = 2 
#NUMBER_ITER = 1
#NUMBER_REPEATS = 1
RANDOM_STATE = 42
GBC_METRIC = 'squared_error'

st.set_page_config(page_title='Car Insurance Fraud Calculator',  layout='wide', page_icon=':Calculator:')

#this is the header
 
t1, t2 = st.columns((0.07,1)) 

t2.title("Car Insurance Fraud Calculator")
t2.markdown("Powered with Machine Learning")

Selector = st.sidebar.selectbox('Select Input Option', ("Choose an option:", 'Database Search', 'Policy Quote','New Customer Search'), placeholder = "Choose an option")

Customer_ID = pd.read_csv("files/policy_number.csv")
Customer_ID = Customer_ID.drop(columns=['Unnamed: 0'])
        
data_df = pd.read_csv("files/insurance_claims.csv")
data_df = data_df.drop(columns=['_c39', 'incident_location', 'policy_bind_date', 'incident_date'])

data_df['fraud_reported'] = data_df['fraud_reported'].str.replace('N','0')
data_df['fraud_reported'] = data_df['fraud_reported'].str.replace('Y','1')

data_df['fraud_reported'] = data_df['fraud_reported'].astype(int)

def machine_learning(cleaned_df, Selected_Customer):

    with st.spinner('Updating Report...'):

        feature_list = list(cleaned_df.columns)

        X = cleaned_df.drop(['fraud_reported'], axis=1).values
        y = cleaned_df['fraud_reported'].values
            
        data = Selected_Customer.drop(['fraud_reported'], axis=1).values

        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

        clf = GradientBoostingClassifier(random_state=RANDOM_STATE,
                                        criterion=GBC_METRIC,
                                        verbose=False)
        
        score = clf.fit(X_train, y_train).predict(data)

        Fraud_risk_test = np.max(clf.predict_proba(data))

        if score==1:
            Fraud_risk_score=Fraud_risk_test

        else:
            Fraud_risk_score=(1-Fraud_risk_test)

        # Get numerical feature importances
        importances = list(clf.feature_importances_)

        # List of tuples with variable and importance
        feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]

        # Sort the feature importances by most important first
        feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)

        #Ten most important features
        ten_most_important = feature_importances[0:10]

        ten_most_important_df = pd.DataFrame(ten_most_important)

        ten_most_important_df.columns = ['Feature', 'Importance']

        ten_most_important_df['Fraud_risk Score'] = Fraud_risk_score

        ten_most_important_df['Claim Accepted?'] = None

        if Fraud_risk_score<=0.68:
            ten_most_important_df['Claim Accepted?'] = ten_most_important_df['Claim Accepted?'].fillna('Yes')
        elif Fraud_risk_score<=0.88:
            ten_most_important_df['Claim Accepted?'] = ten_most_important_df['Claim Accepted?'].fillna('Risky')
        else:
            ten_most_important_df['Claim Accepted?'] = ten_most_important_df['Claim Accepted?'].fillna('No')

    return ten_most_important_df    

if Selector == 'Database Search':

    Customer = st.selectbox('Select Customer', Customer_ID, help = 'Filter report to show only one customer')
    cleaned_df = pd.get_dummies(data_df)

    if Customer:
        Selected_Customer = cleaned_df.loc[cleaned_df['policy_number'] == Customer]
        st.write(Selected_Customer)
    
    g1, g2 = st.columns((1,1))

    global_graph_df = pd.read_csv("files/global_features.csv")
    
    fig = px.bar(global_graph_df, x = 'Feature', y ='Feature Importance')
    
    fig.update_layout(title_text="Global Features Graph",title_x=0,margin= dict(l=0,r=10,b=10,t=30), yaxis_title=None, xaxis_title=None)
    
    g1.plotly_chart(fig, use_container_width=True)

    ten_most_important_df = machine_learning(cleaned_df, Selected_Customer)

    fig2 = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = ten_most_important_df.iat[0,2],
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Fraud Risk Rating", 'font': {'size': 24}},
            gauge = {
                'axis': {'range': [0, 1], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "black"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0.88, 1], 'color': 'red'},
                    {'range': [0.68, 0.88], 'color': 'orange'},
                    {'range': [0, 0.68], 'color': 'green'}],
                'threshold': {
                    'line': {'color': "blue", 'width': 4},
                    'thickness': 0.75,
                    'value': 0.31}}))

    fig2.update_layout(paper_bgcolor = "lavender", font = {'color': "darkblue", 'family': "Arial"})

    g2.plotly_chart(fig2, use_container_width=True) 

elif Selector == 'Policy Quote':

    cleaned_df = data_df.drop(columns=['incident_state','incident_city', 'incident_hour_of_the_day','number_of_vehicles_involved','property_damage','bodily_injuries','witnesses','police_report_available','total_claim_amount','injury_claim','property_claim',
                                    'vehicle_claim','auto_model','auto_year','fraud_reported','authorities_contacted','months_as_customer','collision_type',
                                    'insured_zip', 'incident_type','insured_education_level','insured_occupation','insured_hobbies','insured_relationship',
                                    'incident_severity'])

    #Create number input boxes
    gender = st.sidebar.selectbox('Gender:', ("Choose an option:", 'Female', 'Male'), placeholder = "Choose an option")
    age = st.sidebar.number_input('Age:', value=0, step=1, format="%d")
    policy_state = st.sidebar.selectbox('State', ("Choose an option:", 'OH', 'IL', 'IN'), placeholder = "Choose an option")
    policy_csl = st.sidebar.selectbox('CSL', ("Choose an option:", '100/300', '250/500', '500/1000'), placeholder = "Choose an option")
    policy_deductable = st.sidebar.number_input('policy_deductable:', value=0, step=1, format="%d")
    umbrella_limit = st.sidebar.number_input('Number of umbrella_limit:', value=0, step=1, format="%d")
    capital_gains = st.sidebar.number_input('Capital Gains:')
    capital_loss = st.sidebar.number_input('Capital Loss:')
    auto_make =  st.sidebar.selectbox('Car Manufacturer:', ("Choose an option:", 'Saab', 'Dodge','Suburu','Nissan','Chevrolet','Ford','BMW', 'Toyota','Audi','Accura','Volkswagen','Jeep','Mercedes','Honda'), placeholder = "Choose an option")

    cleaned_df['insured_sex'].iloc[0] = gender
    cleaned_df['age'].iloc[0] = age
    cleaned_df['policy_state'].iloc[0] = policy_state
    cleaned_df['policy_csl'].iloc[0] = policy_csl
    cleaned_df['policy_deductable'].iloc[0] = policy_deductable
    cleaned_df['umbrella_limit'].iloc[0] = umbrella_limit
    cleaned_df['capital-gains'].iloc[0] = capital_gains
    cleaned_df['capital-loss'].iloc[0] = capital_loss
    cleaned_df['auto_make'].iloc[0] = auto_make
    
    if auto_make != 'Choose an option:':

        cleaned_df = pd.get_dummies(cleaned_df)

        Selected_Customer = cleaned_df.iloc[0].to_frame().T

        X = cleaned_df.drop(['policy_annual_premium'], axis=1).values
        y = cleaned_df['policy_annual_premium'].values
            
        data = Selected_Customer.drop(['policy_annual_premium'], axis=1).values

        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

        clf = GradientBoostingRegressor(random_state=RANDOM_STATE,
                                        criterion=GBC_METRIC,
                                        verbose=False)

        score = clf.fit(X_train, y_train).predict(data)[0]  # Assuming the first prediction is the one you want

        # Display the annual car insurance quote using f-string notation
        st.write(f'Your annual car insurance quote is {score:.2f} pounds')

        #score = np.round(score, 2)

        #st.write(f'Your annual car insurance quote is {score:.2f} pounds')

elif Selector == 'New Customer Search':

    cleaned_df = data_df.drop(columns=['incident_state','incident_city', 'incident_hour_of_the_day','property_damage','bodily_injuries','police_report_available','total_claim_amount',
                                    'auto_model','auto_year','fraud_reported','authorities_contacted','months_as_customer',
                                    'insured_zip', 'incident_type','insured_education_level','insured_occupation','insured_relationship',
                                    'policy_state', 'policy_csl', 'policy_deductable', 'umbrella_limit', 'capital-gains', 'capital-loss', 'auto_make'])

    #Create number input boxes
    gender = st.sidebar.selectbox('Gender:', ("Choose an option:", 'Female', 'Male'), placeholder = "Choose an option")
    age = st.sidebar.number_input('Age:', value=0, step=1, format="%d")
    incident_severity = st.sidebar.selectbox('Incident Severity:', ("Choose an option:", 'Trivial Damage', 'Minor Damage', 'Major Damage', 'Total Loss'), placeholder = "Choose an option")
    collision_type = st.sidebar.selectbox('Collision Type:', ("Choose an option:", 'Front Collision', 'Side Collision', 'Rear Collision', 'Other'), placeholder = "Choose an option")
    number_of_vehicles_involved = st.sidebar.number_input('Number of Vehicles Involved:', value=0, step=1, format="%d")
    witnesses = st.sidebar.number_input('Number of Witnesses:', value=0, step=1, format="%d")
    injury_claim = st.sidebar.number_input('Total Injury Claim:')
    property_claim = st.sidebar.number_input('Total Property Claim:')
    vehicle_claim = st.sidebar.number_input('Total Vehicle Claim:')
    insured_hobbies = st.sidebar.selectbox('Insured Hobbies:', ("Choose an option:", 'reading', 'exercise', 'paintball', 'bungie-jumping', 'movies', 'golf', 
                                                                   'camping','kayaking','yachting', 'hiking', 'video-games','skydiving', 'base-jumping', 'board-games',
                                                                    'polo', 'chess', 'dancing', 'sleeping', 'cross-fit', 'basketball'), placeholder = "Choose an option")

    if collision_type == 'Other':
        collision_type = '?'

    cleaned_df = data_df

    cleaned_df['insured_sex'].iloc[531] = gender
    cleaned_df['age'].iloc[531] = age
    cleaned_df['incident_severity'].iloc[531] = incident_severity
    cleaned_df['collision_type'].iloc[531] = collision_type
    cleaned_df['number_of_vehicles_involved'].iloc[531] = number_of_vehicles_involved
    cleaned_df['witnesses'].iloc[531] = witnesses
    cleaned_df['injury_claim'].iloc[531] = injury_claim
    cleaned_df['property_claim'].iloc[531] = property_claim
    cleaned_df['vehicle_claim'].iloc[531] = vehicle_claim
    cleaned_df['insured_hobbies'].iloc[531] = insured_hobbies
    
    if insured_hobbies != 'Choose an option:':

        cleaned_df = pd.get_dummies(cleaned_df)
    
        Selected_Customer = cleaned_df.iloc[531].to_frame().T

        Selected_Customer['total_claim_amount'] = Selected_Customer['injury_claim'] + Selected_Customer['property_claim'] + Selected_Customer['vehicle_claim']

        ten_most_important_df = machine_learning(cleaned_df, Selected_Customer)

        g1, g2 = st.columns((1,1))

        fig = px.bar(ten_most_important_df, x = 'Feature', y='Importance')
            
        fig.update_layout(title_text="Local Features Graph",title_x=0,margin= dict(l=0,r=10,b=10,t=30), yaxis_title=None, xaxis_title=None)
            
        g1.plotly_chart(fig, use_container_width=True)

        fig2 = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = ten_most_important_df.iat[0,2],
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Fraud Risk Rating", 'font': {'size': 24}},
                gauge = {
                    'axis': {'range': [0, 1], 'tickwidth': 1, 'tickcolor': "darkblue"},
                    'bar': {'color': "black"},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0.88, 1], 'color': 'red'},
                        {'range': [0.68, 0.88], 'color': 'orange'},
                        {'range': [0, 0.68], 'color': 'green'}],
                    'threshold': {
                        'line': {'color': "blue", 'width': 4},
                        'thickness': 0.75,
                        'value': 0.31}}))

        fig2.update_layout(paper_bgcolor = "lavender", font = {'color': "darkblue", 'family': "Arial"})

        g2.plotly_chart(fig2, use_container_width=True)     
    
