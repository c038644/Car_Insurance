import pandas as pd
import numpy as np
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import json
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier

def local():

  #Local Features Case for a chosen Selected Customer
  test_df = pd.read_csv("files/P7_test_df.csv")

  Selected_Customer = pd.read_csv("files/selection.csv")

  Selected_Customer = Selected_Customer.drop(columns=['Unnamed: 0'])
  test_df = test_df.drop(columns=['Unnamed: 0'])

  feature_list = list(test_df.columns)

  X = test_df.drop(['TARGET'], axis=1).values
  y = test_df['TARGET'].values

  data = Selected_Customer.drop(['TARGET'], axis=1).values

  X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

  undersample = RandomUnderSampler(sampling_strategy=1)

  X_train, y_train = undersample.fit_resample(X_train, y_train)

  rfc = RandomForestClassifier(max_depth=13, min_samples_leaf=2, min_samples_split=8, n_estimators=552)
  
  score = rfc.fit(X_train, y_train).predict(data)

  Credit_given_test = np.max(rfc.predict_proba(data))

  if score==0:
    credit_score=Credit_given_test

  else:
    credit_score=(1-Credit_given_test)

   # Get numerical feature importances
  importances = list(rfc.feature_importances_)

  # List of tuples with variable and importance
  feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]

  # Sort the feature importances by most important first
  feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)

  #Ten most important features
  ten_most_important = feature_importances[0:10]

  ten_most_important_df = pd.DataFrame(ten_most_important)

  ten_most_important_df.columns = ['Feature', 'Importance']

  ten_most_important_df['Credit Score'] = credit_score

  ten_most_important_df['Credit Granted?'] = None

  if credit_score>=0.35:
    ten_most_important_df['Credit Granted?'] = ten_most_important_df['Credit Granted?'].fillna('Yes')
  elif credit_score>=0.25:
    ten_most_important_df['Credit Granted?'] = ten_most_important_df['Credit Granted?'].fillna('Risky')
  else:
    ten_most_important_df['Credit Granted?'] = ten_most_important_df['Credit Granted?'].fillna('No')

  ten_most_important_df.to_csv("files/Customer_score.csv")

  print('Customer Score Ready')
  return(ten_most_important_df)


def global_data():

  test_df = pd.read_csv("files/P7_test_df.csv")

  #Global Features Case

  feature_list = list(test_df.columns)

  X = test_df.drop(['TARGET'], axis=1).values
  y = test_df['TARGET'].values

  X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

  undersample = RandomUnderSampler(sampling_strategy=1)

  X_train, y_train = undersample.fit_resample(X_train, y_train)

  rfc = RandomForestClassifier(max_depth=13, min_samples_leaf=2, min_samples_split=8, n_estimators=552)

  rfc.fit(X_train, y_train)

  # Get numerical feature importances
  importances = list(rfc.feature_importances_)

  # List of tuples with variable and importance
  feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]

  # Sort the feature importances by most important first
  feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)

  #Ten most important features
  ten_most_important = feature_importances[0:10]

  Global_Features = pd.DataFrame(ten_most_important)

  Global_Features.columns = ['Feature', 'Importance']

  Global_Features.to_csv("files/Global_Features.csv")

  # Print out the feature and importances 
  return(Global_Features)

st.set_page_config(page_title='Credit Rating Calculator',  layout='wide', page_icon=':Calculator:')

#this is the header
 
t1, t2 = st.columns((0.07,1)) 

t2.title("Credit Rating Calculator")
t2.markdown("with Global and Local Customer Data")


with st.spinner('Updating Report...'):
    
    Customer_ID = pd.read_csv("files/policy_number.csv")
    Customer_ID = Customer_ID.drop(columns=['Unnamed: 0'])
    
    data_df = pd.read_csv("files/insurance_claims.csv")
    data_df = data_df.drop(columns=['_c39', 'incident_location', 'policy_bind_date', 'incident_date'])

    data_df['fraud_reported'] = data_df['fraud_reported'].str.replace('N','0')
    data_df['fraud_reported'] = data_df['fraud_reported'].str.replace('Y','1')

    data_df['fraud_reported'] = data_df['fraud_reported'].astype(int)

    cleaned_df = pd.get_dummies(data_df)

    Customer = st.selectbox('Select Customer', Customer_ID, help = 'Filter report to show only one customer')


    if Customer:
        Selected_Customer = data_df.loc[data_df['policy_number'] == Customer]
        st.write(Selected_Customer)
        Selected_Customer.to_csv("files/selection.csv")
        #local = requests.get("local").json()
        #st.json(local) 
    
    local()

    g1, g2, g3 = st.columns((1,1,1))

    local_graph_df = pd.read_csv("files/Customer_score.csv")
    
    fig = px.bar(local_graph_df, x = 'Feature', y='Importance')
    
 
    fig.update_layout(title_text="Local Features Graph",title_x=0,margin= dict(l=0,r=10,b=10,t=30), yaxis_title=None, xaxis_title=None)
    
    g1.plotly_chart(fig, use_container_width=True)

    fig2 = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = local_graph_df.iat[0,3],
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Credit Rating", 'font': {'size': 24}},
        gauge = {
            'axis': {'range': [0, 1], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "black"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 0.26], 'color': 'red'},
                {'range': [0.26, 0.36], 'color': 'orange'},
                {'range': [0.36, 1], 'color': 'green'}],
            'threshold': {
                'line': {'color': "blue", 'width': 4},
                'thickness': 0.75,
                'value': 0.31}}))

    fig2.update_layout(paper_bgcolor = "lavender", font = {'color': "darkblue", 'family': "Arial"})

    g2.plotly_chart(fig2, use_container_width=True) 

    #dashboard = requests.get("global_data").json()

    #global_data() 

    global_graph_df = pd.read_csv("files/Global_Features.csv")

    global_graph_df = global_graph_df.drop(columns=['Unnamed: 0'])
    
    fig = px.bar(global_graph_df, x = 'Feature', y='Importance')
    
    fig.update_layout(title_text="Global Features Graph",title_x=0,margin= dict(l=0,r=10,b=10,t=30), yaxis_title=None, xaxis_title=None)
    
    g3.plotly_chart(fig, use_container_width=True)

    Selected_Customer = pd.read_csv("files/selection.csv")

    Feature_List = pd.read_csv("files/P7_Features.csv")

    Feature = st.selectbox('Select Feature', Feature_List, help = 'Filter report to show only one feature')

    Selected_Feature = all_data.loc[all_data[Feature] == Feature].any()

    g4, g5 = st.columns((1,2))

    fig = px.scatter(Selected_Customer, x = 'SK_ID_CURR', y = Feature)
    
    fig.update_layout(title_text="Local Feature Graph",title_x=0,margin= dict(l=0,r=10,b=10,t=30), yaxis_title=None, xaxis_title=None)
    
    g4.plotly_chart(fig, use_container_width=True)

    fig = px.scatter(all_data, x = 'SK_ID_CURR', y = Feature)
    
    fig.update_layout(title_text="Global Feature Graph",title_x=0,margin= dict(l=0,r=10,b=10,t=30), yaxis_title=None, xaxis_title=None)
    
    g5.plotly_chart(fig, use_container_width=True)
