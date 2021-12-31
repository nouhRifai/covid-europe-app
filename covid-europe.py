import streamlit as st
import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingClassifier


COUNTRY_MEASURES_DATASET = 'https://www.ecdc.europa.eu/sites/default/files/documents/response_graphs_data_2021-12-09.csv'
DAILY_CASES_DATASET = 'https://opendata.ecdc.europa.eu/covid19/nationalcasedeath_eueea_daily_ei/csv'
AGE_SPECIFIC_CASES_DATASET = 'https://opendata.ecdc.europa.eu/covid19/agecasesnational/csv/data.csv'
TESTING_DATASET = 'https://opendata.ecdc.europa.eu/covid19/testing/csv/data.csv'
HOSPITAL_ICU_ADMISSION = 'https://opendata.ecdc.europa.eu/covid19/hospitalicuadmissionrates/csv/data.csv'
countries = [
    '0 - Austria',
    '1 - Belgium',
    '2 - Croatia',
    '3 - Cyprus',
    '4 - Czechia',
    '5 - Denmark',
    '6 - Estonia',
    '7 - Finland',
    '8 - France',
    '9 - Germany',
    '10 - Greece',
    '11 - Hungary',
    '12 - Iceland',
    '13 - Ireland',
    '14 - Italy',
    '15 - Latvia',
    '16 - Liechtenstein',
    '17 - Lithuania',
    '18 - Luxembourg',
    '19 - Malta',
    '20 - Netherlands',
    '21 - Norway',
    '22 - Poland',
    '23 - Portugal',
    '24 - Romania',
    '25 - Slovakia',
    '26 - Slovenia',
    '27 - Spain',
    '28 - Sweden',
    '29 - ALL COUNTRIES'
    ]

age_groups = [
    '0 - Under 15yo',
    '1 - Between 15 and 24yo',
    '2 - Between 25 and 49yo',
    '3 - Between 50 and 64yo',
    '4 - Between 65 and 79yo',
    '5 - Above 80yo'
]

risk_dict = {0:{
                  'color' : 'green',
                  'color_code' : '#00FF00',
                  'daily_cases_per_capita_risk' : 'No Risk',
                  'positivity_rate' : 'Adequate Testing',
                },
              1:{
                  'color' : 'yellow',
                  'color_code' : '#FFD700',
                  'daily_cases_per_capita_risk' : 'Low Risk',
                  'positivity_rate' : 'High Rate of Testing',
                },
              2:{
                  'color': 'orange',
                  'color_code' : '#FF4500',
                  'daily_cases_per_capita_risk' : 'Medium Risk',
                  'positivity_rate' : 'Low Rate of Testing',
                },
              3:{
                  'color': 'red',
                  'color_code' : '#FF0000',
                  'daily_cases_per_capita_risk' : 'High Risk',
                  'positivity_rate' : 'Inadequate Testing',
                },
              4:{
                  'color': 'crimson',
                  'color_code' : '#DC143C',
                  'daily_cases_per_capita_risk' : 'Severe Outbreak',
                }
              }

st.set_page_config(layout="wide")
def intro_texts():
    st.header('Covid-19 in european countries')
    st.subheader('Before we start !')
    st.markdown('''
This work was realised by :

- Nour-El Houda RAYAD (rayadnourelhouda@gmail.com)

- Nouh RIFAI (gr.nooh@gmail.com)


This work has an educational purpose, and it is debatable in some perspectives. 

So, if there is any question or remark, we would like you to share it with us in order to improve our work!
    ''')
    #################################
    st.subheader('Introduction')
    st.markdown('''
        Coronaviruses are a large family of viruses that are known to cause illness ranging from the common cold to more severe diseases such as Middle East Respiratory Syndrome (MERS) and Severe Acute Respiratory Syndrome (SARS).
        
        A novel coronavirus (COVID-19) was identified in 2019 in Wuhan, China. This is a new coronavirus that has not been previously identified in humans.
    ''')
    #################################
    st.subheader('Problem to solve')
    st.markdown('''
        While the virus is causing huge amount of causalties, many decisions have been taken to limit its spread. However, in lack of solid tools and analysis, these decisions tend to have a low impact in limiting the pandemic situation's growth.
        
        So how can we develop tools, in order to improve this decisional task and achieve better results?
    ''')
    #################################
    st.subheader('Our Solution')
    st.markdown('''
        Technology has advanced in an enourmous way, and for many years it has served its main purpose 'to benefit the lives of mankind'. Though, recently, it's facing huge difficulties to eradicate this pandemic situation. 
        
        Artificial Intelligence is on top of technologies that were used to limite, reduce or even eradicate the spread of the virus. 
        
        Among the major applications of ai in this matter, is predicting new cases, predicting test positivity rate, or classifying risk.
        
        We will cover in our solution :
        
        - The covid new cases predictor; (Regression model)

        - The daily cases per capita risk classifier; (Classification model)

        - The test positivity rate classifier. (Classification model)

        Please note that the current page contains only data visualizations, and the predictors!\n 
    ''')
    st.write("The complete work is available in this [Git repo](https://github.com/nouhRifai/covid-europe).\n")
    #################################
    st.subheader('Content')
    st.markdown('''

        - Data;

        - Predictors;
        
        - Data Visualizations (Available in next version).

    ''')
    ##################################
    st.subheader('Data:\n')
    st.write('''First of all, we collected data from the [European Centre for Disease Prevention and Control](https://opendata.ecdc.europa.eu/) website. We used only 5 datasets for our project. 
    The five datasets included in this project are :

- DAILY CASES DATASET
- AGE SPECIFIC CASES DATASET
- COUNTRY MEASURES DATASET
- TESTING DATASET
- HOSPITAL ICU ADMISSION DATASET

    Please note that the website provides with each dataset a descriptive file.\n''')



# '''
# --------------------------------------------------------------------------------------------------------------------------------------------
# Importing Data and Preparations
# --------------------------------------------------------------------------------------------------------------------------------------------
# '''

def preprocess_daily_cases_dataset(daily_cases):
    daily_cases['dateRep'] = pd.to_datetime(daily_cases['dateRep'], format='%d/%m/%Y')
    def set_year_week(x): 
        my_zero = ''
        if x['dateRep'].week <10:
            my_zero = '0'
        return str(x['dateRep'].year) + '-W' + my_zero + str(x['dateRep'].week)
    #apply the function on the dataset and creating year_week column
    daily_cases['year_week'] = daily_cases.apply(set_year_week , axis=1)
    #apply the value Nan to all the negative value of cases
    daily_cases.loc[daily_cases['cases'] <0, 'cases'] = daily_cases[daily_cases['cases']< 0]['cases'].apply(lambda x : np.NaN)
    daily_cases.loc[daily_cases['deaths'] <0, 'deaths'] = daily_cases[daily_cases['deaths']< 0]['deaths'].apply(lambda x : np.NaN)
    daily_cases = daily_cases.sort_values(['countriesAndTerritories','dateRep']).fillna(method='ffill')
    daily_cases = daily_cases[daily_cases['dateRep']>'2021-03-01']
    _ = daily_cases.replace(to_replace=0,method='ffill',inplace=True)
    daily_cases = daily_cases.groupby(['countriesAndTerritories','year_week'])['cases','deaths'].agg(['sum','max','min','mean']).reset_index()
    return daily_cases

def preprocess_age_specific_cases_dataset(age_specific_cases):
    age_specific_cases['year_week'] = age_specific_cases['year_week'].apply(lambda x: x.split('-')[0]+'-W'+x.split('-')[1])
    age_specific_cases=age_specific_cases[age_specific_cases['year_week']>='2020-W15']
    no_bulgaria_age_specific_cases = age_specific_cases[age_specific_cases['country']!= 'Bulgaria'] 
    values = {'new_cases' : 0, 'rate_14_day_per_100k': 0}
    no_bulgaria_age_specific_cases[no_bulgaria_age_specific_cases['country']=='Lithuania'] = no_bulgaria_age_specific_cases[no_bulgaria_age_specific_cases['country']=='Lithuania'].fillna(value=values) 
    no_bulgaria_age_specific_cases = no_bulgaria_age_specific_cases.fillna(method="ffill")
    age_specific_cases = no_bulgaria_age_specific_cases
    cleanup_age_group = {"age_group": {"<15yr": 0,
                                    "15-24yr": 1,
                                    "25-49yr": 2,
                                    "50-64yr": 3,
                                    "65-79yr": 4,
                                    "80+yr": 5,
                                    }
                        }
    age_specific_cases = age_specific_cases.replace(cleanup_age_group)
    return age_specific_cases

def preprocess_testing_dataset(testing):
    testing = testing.fillna(method='ffill')
    testing = testing.groupby(['country', 'year_week'])['new_cases','tests_done','testing_rate','positivity_rate'].agg(['sum','max','min','mean']).reset_index()
    return testing

def preprocess_hospital_icu_admission_dataset(hospital_icu_admission):
    hospital_icu_admission = hospital_icu_admission.groupby(['country', 'year_week'])['value'].agg(['sum','max','min','mean']).reset_index().rename(columns={'sum':'admission_total','max':'admission_max','min':'admission_min','mean':'admission_mean'})
    return hospital_icu_admission

def preprocess_country_measures_dataset(country_measures):
    country_measures['date_start'] = pd.to_datetime(country_measures['date_start'], format = '%Y-%m-%d')
    country_measures['date_end'] = pd.to_datetime(country_measures['date_end'], format = '%Y-%m-%d')
    country_measures['date_end'] = country_measures['date_end'].fillna(pd.to_datetime("today"))
    country_measures = country_measures.reset_index()
    def set_country_measures_year_week (date_start, date_end):
        year_week = []
        while date_start < date_end:
            my_zero=''
            if date_start.week < 10:
                my_zero='0'
            year_week.append(str(date_start.year)+'-W'+my_zero+str(date_start.week))
            date_start = date_start + pd.Timedelta('7 days')
        #print(year_week)
        return year_week
    country_measures['year_week'] = country_measures.apply(lambda x: set_country_measures_year_week(x['date_start'],x['date_end']),axis=1)
    helper_df = country_measures.apply(lambda x : pd.Series(x['year_week']),axis=1)
    helper_df[['index','Country','Response_measure']] = country_measures[['index','Country','Response_measure']]
    year_week_country_measures = helper_df.set_index(['index','Country','Response_measure']).stack().reset_index()[['Country','Response_measure',0]].rename(columns={0:'year_week'})
    with open('country_measures.json') as json_file:
        new_columns_dict = json.load(json_file)
        for column_name in new_columns_dict.keys():
            current_column = new_columns_dict[column_name]
            def set_new_column_values(x):
                if x['Response_measure'] in current_column.keys():
                    return current_column[x['Response_measure']]
                else:
                    return 0
            year_week_country_measures[column_name] = year_week_country_measures.apply(set_new_column_values,axis=1)
    country_measures = year_week_country_measures.groupby(['Country','year_week'])[list(new_columns_dict.keys())].agg('max').reset_index()
    return country_measures

def preprocess_data(daily_cases, age_specific_cases, testing, hospital_icu_admission, country_measures):
    p_daily_cases = preprocess_daily_cases_dataset(daily_cases)
    p_age_specific_cases = preprocess_age_specific_cases_dataset(age_specific_cases)
    p_testing = preprocess_testing_dataset(testing)
    p_hospital_icu_admission = preprocess_hospital_icu_admission_dataset(hospital_icu_admission)
    p_country_measures = preprocess_country_measures_dataset(country_measures)
    return (p_daily_cases, p_age_specific_cases, p_testing, p_hospital_icu_admission, p_country_measures)

def load_data():
    daily_cases = pd.read_csv(DAILY_CASES_DATASET)
    age_specific_cases = pd.read_csv(AGE_SPECIFIC_CASES_DATASET)
    testing = pd.read_csv(TESTING_DATASET)
    hospital_icu_admission = pd.read_csv(HOSPITAL_ICU_ADMISSION)
    country_measures = pd.read_csv(COUNTRY_MEASURES_DATASET)
    return (daily_cases, age_specific_cases, testing, hospital_icu_admission, country_measures)

def get_color_code(index):
    global risk_dict
    return risk_dict[index].color_code
def get_positivity_rate_risk(index):
    global risk_dict
    return risk_dict[index].positivity_rate
def get_daily_cases_per_capita_risk(index):
    global risk_dict
    return risk_dict[index].daily_cases_per_capita_risk


def last_prep(without_dc_df):
    
    def daily_cases_per_capita_risk(x):
        if x < 1:
            return 0
        elif x < 10:
            return 1
        elif x < 25:
            return 2
        elif x < 75:
            return 3
        else:
            return 4

    def positivity_rate(x):
        if x < 3:
            return 0
        elif x < 10:
            return 1
        elif x < 20:
            return 2
        else:
            return 3

    without_dc_df['daily_cases_per_capita_risk'] = without_dc_df['rate_14_day_per_100k'].apply(daily_cases_per_capita_risk)
    without_dc_df['positivity_rate'] = without_dc_df['positivity_rate_mean'].apply(positivity_rate)
    columns_to_keep = ['country','year_week','age_group','new_cases','rate_14_day_per_100k','population','SHM','SMHT','EINCM','EIPCM','EISCM','EIHCM','EICMT','MGEM','MGIM','MGOM','MGSO50M','MGSO100M','MGSO500M','MGSO1000M','MGMT','WM','WMT','TWM','TWMT','MOM','MOMT','MIM','MIMT','EVCM','EVCMT','PTCM','PTCMT','GCM','GCMT','HACM','HACMT','NESCM','NESCMT','POWCM','POWCMT','RCCM','RCCMT','SCM','SCMT','PGRM','PGRMT','testing_rate_sum','testing_rate_max','testing_rate_min','testing_rate_mean','positivity_rate_sum','positivity_rate_max','positivity_rate_min','positivity_rate_mean','admission_total','admission_max','admission_min','admission_mean','daily_cases_per_capita_risk','positivity_rate']
    without_dc_df = without_dc_df[columns_to_keep]
    country_encoder = LabelEncoder()
    country_encoder.fit(without_dc_df['country'])
    without_dc_df['country'] = country_encoder.transform(without_dc_df['country'])
    without_dc_df['year'] = without_dc_df['year_week'].apply(lambda x: int(x[0:4]))
    without_dc_df['week'] = without_dc_df['year_week'].apply(lambda x: int(x[6:8]))
    without_dc_df = without_dc_df.drop(['year_week'],axis=1)
    without_dc_df = without_dc_df.sort_values(['country','age_group'])
    df = pd.DataFrame()
    column_names=['new_cases-1','new_cases-2','new_cases-3','new_cases-4','new_cases-5','new_cases-6',
                'rate_14_day_per_100k-1','rate_14_day_per_100k-2','rate_14_day_per_100k-3','rate_14_day_per_100k-4','rate_14_day_per_100k-5','rate_14_day_per_100k-6',
                'testing_rate_mean-1','testing_rate_mean-2','testing_rate_mean-3','testing_rate_mean-4','testing_rate_mean-5','testing_rate_mean-6',
                'positivity_rate_mean-1','positivity_rate_mean-2','positivity_rate_mean-3','positivity_rate_mean-4','positivity_rate_mean-5','positivity_rate_mean-6',
                'admission_mean-1','admission_mean-2','admission_mean-3','admission_mean-4','admission_mean-5','admission_mean-6',
                'daily_cases_per_capita_risk-1','daily_cases_per_capita_risk-2','daily_cases_per_capita_risk-3','daily_cases_per_capita_risk-4','daily_cases_per_capita_risk-5','daily_cases_per_capita_risk-6',
                'positivity_rate-1','positivity_rate-2','positivity_rate-3','positivity_rate-4','positivity_rate-5','positivity_rate-6',
                ]
    for column_name in column_names:
        new_col=[]
        limit = int(column_name[-1])
        col = column_name[0:-2]
        for country in range(0,29):
            new_subset=[]
            for age_group in range(0,6):
                new_subset.extend([np.nan for i in range(0,limit)])
                new_subset.extend(without_dc_df[(without_dc_df['country']==country)&(without_dc_df['age_group']==age_group)][col].iloc[0:-limit].to_list())
            new_col.extend(new_subset)
        df[column_name] = new_col
    age_group_column = without_dc_df['age_group'].to_list()
    df['country'] = without_dc_df['country']
    df['age_group'] = age_group_column
    other_columns = ['population','year','week','SHM','SMHT','EINCM','EIPCM','EISCM','EIHCM','EICMT','MGEM','MGIM','MGOM','MGSO50M','MGSO100M','MGSO500M','MGSO1000M','MGMT','WM','WMT','TWM','TWMT','MOM','MOMT','MIM','MIMT','EVCM','EVCMT','PTCM','PTCMT','GCM','GCMT','HACM','HACMT','NESCM','NESCMT','POWCM','POWCMT','RCCM','RCCMT','SCM','SCMT','PGRM','PGRMT','new_cases','rate_14_day_per_100k','daily_cases_per_capita_risk','positivity_rate','positivity_rate','testing_rate_mean','positivity_rate_mean','admission_mean']
    other_df_dict = without_dc_df[other_columns].to_dict()
    other_df = pd.DataFrame(other_df_dict)
    other_df = other_df.reset_index()
    indices = other_df['index'].to_list()
    df['index'] = indices
    df = df.merge(other_df,on='index')
    df = df.fillna(method='ffill')
    df = df.fillna(method='bfill')
    return df,country_encoder


def merge_data(daily_cases, age_specific_cases, testing, hospital_icu_admission, country_measures):
    #Preparing for merge
    ##Daily Cases
    daily_cases.columns = ['_'.join(col) for col in daily_cases.columns]
    daily_cases = daily_cases.rename(columns={'countriesAndTerritories_':'country', 'year_week_':'year_week'})
    ##Age Specific
    columns_to_keep = age_specific_cases.columns.to_list()[0:-1]
    age_specific_cases = age_specific_cases[columns_to_keep]
    ##Testing
    testing.columns = ['_'.join(col) for col in testing.columns]
    testing = testing.rename(columns={'country_':'country', 'year_week_':'year_week'})
    ##Country Measures
    country_measures = country_measures.rename(columns={'Country':'country'})
    #Applying merge
    ##with daily cases
    with_dc_df = daily_cases.merge(age_specific_cases, on=['country','year_week']).merge(country_measures, on=['country','year_week']).merge(testing, on=['country','year_week']).merge(hospital_icu_admission, on=['country','year_week'])
    ##without daily cases
    without_dc_df = age_specific_cases.merge(country_measures, on=['country','year_week']).merge(testing, on=['country','year_week']).merge(hospital_icu_admission, on=['country','year_week'])
    return (with_dc_df, without_dc_df)

def get_models(df):
    X = df.drop(['new_cases','rate_14_day_per_100k','daily_cases_per_capita_risk','positivity_rate','testing_rate_mean','positivity_rate_mean','admission_mean'],axis=1)
    #for regression
    y1 = df['new_cases']
    #for classification
    y2 = df['daily_cases_per_capita_risk'].astype('int')
    y3 = df['positivity_rate'].astype('int')
    X_train, X_test, y1_train, y1_test = train_test_split(X,y1,random_state=0)
    GBR = GradientBoostingRegressor(learning_rate= 0.1,
                                    max_depth= 10,
                                    min_impurity_decrease= 0.01,
                                    n_estimators= 500,
                                    n_iter_no_change= 10)
    GBR.fit(X_train,y1_train)

    X_train, X_test, y2_train, y2_test = train_test_split(X,y2,random_state=0)
    GBC1 = GradientBoostingClassifier(learning_rate= 0.1,
                                    max_depth= 10,
                                    min_impurity_decrease= 0.01,
                                    n_estimators= 500,
                                    n_iter_no_change= 10)
    GBC1.fit(X_train,y2_train)

    X_train, X_test, y3_train, y3_test = train_test_split(X,y3,random_state=0)
    GBC2 = GradientBoostingClassifier(learning_rate= 0.1,
                                    max_depth= 10,
                                    min_impurity_decrease= 0.01,
                                    n_estimators= 500,
                                    n_iter_no_change= 10)
    GBC2.fit(X_train,y3_train)

    return GBR, GBC1, GBC2

@st.cache(ttl=60*60*5)
def preparations():
    daily_cases, age_specific_cases, testing, hospital_icu_admission, country_measures = load_data()
    p_daily_cases, p_age_specific_cases, p_testing, p_hospital_icu_admission, p_country_measures = preprocess_data(daily_cases, age_specific_cases, testing, hospital_icu_admission, country_measures)
    data1, data2 = merge_data(p_daily_cases, p_age_specific_cases, p_testing, p_hospital_icu_admission, p_country_measures)
    data2,country_encoder = last_prep(data2)
    GBR, GBC1, GBC2 = get_models(data2)
    return (GBR, GBC1, GBC2, data2, country_encoder)

# '''
# --------------------------------------------------------------------------------------------------------------------------------------------
# First func:
#     Input : country, age_group (year_week = next week)
#     Output : predicted_new_cases, predicted_daily_cases_per_capita_risk, predicted_test_positivity_rate

# Second func:
#     Input : age_group
#     Output : in every country (predicted_new_cases, predicted_daily_cases_per_capita_risk, predicted_test_positivity_rate)
# --------------------------------------------------------------------------------------------------------------------------------------------
# '''

def get_previous_data1(country, age_group, year, week, df):
    final_columns = df.drop(['new_cases','rate_14_day_per_100k','daily_cases_per_capita_risk','positivity_rate','testing_rate_mean','positivity_rate_mean','admission_mean'],axis=1).columns.to_list()
    df = df.drop(['new_cases-6','rate_14_day_per_100k-6','testing_rate_mean-6','positivity_rate_mean-6','admission_mean-6','daily_cases_per_capita_risk-6','positivity_rate-6'],axis=1)
    c_columns = ['new_cases','rate_14_day_per_100k','testing_rate_mean','positivity_rate_mean','admission_mean','daily_cases_per_capita_risk','positivity_rate']
    X = pd.DataFrame()
    columns_names = df.columns.to_list()
    for cn in columns_names:
        if not cn[-1].isdigit():
            if cn in c_columns:
                column_name = cn + '-1'
                X[column_name]=[df[(df['country']==country)&(df['age_group']==age_group)].iloc[-1][cn]]
                continue
            X[cn] = [df[(df['country']==country)&(df['age_group']==age_group)].iloc[-1][cn]]
            continue
        else:
            column_name = cn[:-2] + '-' + str(int(cn[-1])+1)
            X[column_name] = [df[(df['country']==country)&(df['age_group']==age_group)].iloc[-1][cn]]
    return X[final_columns]


def get_previous_data(age_group, year, week, df):
    final_columns = df.drop(['new_cases','rate_14_day_per_100k','daily_cases_per_capita_risk','positivity_rate','testing_rate_mean','positivity_rate_mean','admission_mean'],axis=1).columns.to_list()
    df = df.drop(['new_cases-6','rate_14_day_per_100k-6','testing_rate_mean-6','positivity_rate_mean-6','admission_mean-6','daily_cases_per_capita_risk-6','positivity_rate-6'],axis=1)
    c_columns = ['new_cases','rate_14_day_per_100k','testing_rate_mean','positivity_rate_mean','admission_mean','daily_cases_per_capita_risk','positivity_rate']
    X_final = pd.DataFrame(columns=final_columns)
    columns_names = df.columns.to_list()
    for country in range(0,29):
        X = pd.DataFrame()
        for cn in columns_names:
            if not cn[-1].isdigit():
                if cn in c_columns:
                    column_name = cn + '-1'
                    X[column_name]=[df[(df['country']==country)&(df['age_group']==age_group)].iloc[-1][cn]]
                    continue
                X[cn] = [df[(df['country']==country)&(df['age_group']==age_group)].iloc[-1][cn]]
                continue
            else:
                column_name = cn[:-2] + '-' + str(int(cn[-1])+1)
                X[column_name] = [df[(df['country']==country)&(df['age_group']==age_group)].iloc[-1][cn]]
        X = X[final_columns]
        X_final = X_final.append(X)
    return X_final[final_columns]
def get_predictions(country, age_group, new_cases_predictor, daily_cases_per_capita_risk_predictor, test_positivity_rate_predictor, df):
    year = pd.to_datetime('today').year
    week = pd.to_datetime('today').week
    country_code = int(country.split(' - ')[0])
    country_name = country.split(' - ')[1]
    age_group_code = int(age_group.split(' - ')[0])
    age_group_name = age_group.split(' - ')[1]

    if int(country_code) == 29:
        X = get_previous_data(age_group_code, year, week, df)
        return 0,(new_cases_predictor.predict(X), X['new_cases-1'].to_list()), daily_cases_per_capita_risk_predictor.predict(X), test_positivity_rate_predictor.predict(X)
    else:
        X = get_previous_data1(country_code, age_group_code, year, week, df)
        return 0,(new_cases_predictor.predict(X), X['new_cases-1'].to_list()), daily_cases_per_capita_risk_predictor.predict(X), test_positivity_rate_predictor.predict(X)


# '''
# --------------------------------------------------------------------------------------------------------------------------------------------
# Main App interface
# --------------------------------------------------------------------------------------------------------------------------------------------
# '''

def main_app():
    intro_texts()
    #################################
    #Preparing data first
    data_load_state = st.text('Loading data...')
    
    new_cases_predictor, daily_cases_per_capita_risk_predictor, test_positivity_rate_predictor, df, country_encoder = preparations()
    data_load_state.text('Done!')
    #Showing predictor options
    st.subheader('Predictor:\n')
    col1, col2= st.columns(2)
    country_choice = col1.selectbox('Selected Country:', countries)
    age_group_choice = col2.selectbox('Selected Age Group:', age_groups)
    results_type, predicted_new_cases, predicted_daily_cases_per_capita_risk, predicted_test_positivity_rate= get_predictions(country_choice, age_group_choice, new_cases_predictor, daily_cases_per_capita_risk_predictor, test_positivity_rate_predictor, df)
    col41, col42, col43, col44 = st.columns(4)
    if country_choice != '29 - ALL COUNTRIES':
        col41.metric(label='',value=country_choice.split(' - ')[1])
        col42.metric(label="New cases for people {}".format(age_group_choice.split(' - ')[1]), value=round(predicted_new_cases[0][0]), delta = round(predicted_new_cases[0][0])-round(predicted_new_cases[1][0]), delta_color="inverse")
        col43.metric(label="Daily Cases per 100k risk for peolple {}".format(age_group_choice.split(' - ')[1]), value=risk_dict[predicted_daily_cases_per_capita_risk[0]]['daily_cases_per_capita_risk'])
        col44.metric(label="Positivity rate", value=risk_dict[predicted_test_positivity_rate[0]]['positivity_rate'])
    else:
        for i in range(0,len(predicted_new_cases[1])):
            # with st.container():
            with st.expander(country_encoder.inverse_transform([i])[0]):
                col41, col42, col43, col44 = st.columns(4)
                col41.metric(label='',value=country_encoder.inverse_transform([i])[0])
                col42.metric(label="New cases for people {}".format(age_group_choice.split(' - ')[1]), value=round(predicted_new_cases[0][i]), delta = round(predicted_new_cases[0][i])-round(predicted_new_cases[1][i]), delta_color="inverse")
                col43.metric(label="Daily Cases per 100k risk for peolple {}".format(age_group_choice.split(' - ')[1]), value=risk_dict[predicted_daily_cases_per_capita_risk[i]]['daily_cases_per_capita_risk'])
                col44.metric(label="Positivity rate", value=risk_dict[predicted_test_positivity_rate[i]]['positivity_rate'])
        #################################
        # st.subheader('Data Visualizations:\n')


main_app()