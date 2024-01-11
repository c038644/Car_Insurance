import pandas as pd
import numpy as np
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
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

Selector = st.sidebar.selectbox('Select Input Option', ("Choose an option:", 'Database Search', 'New Customer Search'), placeholder = "Choose an option")

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
                                        verbose=False,
                                        max_depth=45,
                                        max_features='sqrt',
                                        min_samples_leaf=2,
                                        min_samples_split=2,
                                        n_estimators=836)
        # Best Custom Score: 0.7853282025046155

        # Hyperparameter Tuning: {'clf__max_depth': 45, 'clf__max_features': 'sqrt', 'clf__min_samples_leaf': 2, 'clf__min_samples_split': 2, 'clf__n_estimators': 836}

        score = clf.fit(X_train, y_train).predict(data)

        Fraud_risk_test = np.max(clf.predict_proba(data))
        st.write(Fraud_risk_test)

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

elif Selector == 'New Customer Search':
        
    #Create number input boxes
    gender = st.sidebar.selectbox('Select Input Option', ("Choose an option:", 'Female', 'Male'), placeholder = "Choose an option")
    age = st.sidebar.number_input('Age:')
    #incident_city = st.sidebar.text_input('Incident City:')
    incident_severity = st.sidebar.selectbox('Select Input Option', ("Choose an option:", 'Trivial Damage', 'Minor Damage', 'Major Damage', 'Total Loss'), placeholder = "Choose an option")
    collision_type = st.sidebar.selectbox('Select Input Option', ("Choose an option:", 'Front Collision', 'Side Collision', 'Rear Collision', '?'), placeholder = "Choose an option")
    number_of_vehicles_involved = st.sidebar.number_input('Number of Vehicles Involved:')
    witnesses = st.sidebar.number_input('Number of Witnesses:')
    injury_claim = st.sidebar.number_input('Total Injury Claim:')
    property_claim = st.sidebar.number_input('Total Property Claim:')
    vehicle_claim = st.sidebar.number_input('Total Vehicle Claim:')
    insured_hobbies = st.sidebar.selectbox('Select Input Option', ("Choose an option:", 'reading', 'exercise', 'paintball', 'bungie-jumping', 'movies', 'golf', 
                                                                   'camping','kayaking','yachting', 'hiking', 'video-games','skydiving', 'base-jumping', 'board-games',
                                                                    'polo', 'chess', 'dancing', 'sleeping', 'cross-fit', 'basketball'), placeholder = "Choose an option")

    # Get the number of rows in the DataFrame
    num_rows = len(data_df)

    # Generate a random index within the range of the number of rows
    random_index = random.randint(0, num_rows - 1)
    #Selected_Customer = cleaned_df.sample(n=1)

    #if gender == 'Female':
    #    Selected_Customer['insured_sex_FEMALE'] = 1
    #    Selected_Customer['insured_sex_MALE'] = 0
    #else:
    #    Selected_Customer['insured_sex_FEMALE'] = 0
    #    Selected_Customer['insured_sex_MALE'] = 1    

    data_df['insured_sex'].iloc[random_index] = gender
    data_df['age'].iloc[random_index] = age
    data_df['incident_severity'].iloc[random_index] = incident_severity
    data_df['collision_type'].iloc[random_index] = collision_type
    data_df['number_of_vehicles_involved'].iloc[random_index] = number_of_vehicles_involved
    data_df['witnesses'].iloc[random_index] = witnesses
    data_df['injury_claim'].iloc[random_index] = injury_claim
    data_df['property_claim'].iloc[random_index] = property_claim
    data_df['vehicle_claim'].iloc[random_index] = vehicle_claim
    data_df['insured_hobbies'].iloc[random_index] = insured_hobbies
    
    
        
    #Selected_Customer['incident_severity'] = incident_severity
    #Selected_Customer['collision_type'] = collision_type
    #Selected_Customer['number_of_vehicles_involved'] = number_of_vehicles_involved
    #Selected_Customer['witnesses'] = witnesses
    #Selected_Customer['injury_claim'] = injury_claim
    #Selected_Customer['property_claim'] = property_claim
    #Selected_Customer['vehicle_claim'] = vehicle_claim
    #Selected_Customer['insured_hobbies'] = insured_hobbies

    Selected_Customer = data_df.iloc[random_index]
    st.write(Selected_Customer)

    if insured_hobbies:

        Selected_Customer['total_claim_amount'] = Selected_Customer['injury_claim'] + Selected_Customer['property_claim'] + Selected_Customer['vehicle_claim']

        cleaned_df = pd.get_dummies(data_df)

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

    
