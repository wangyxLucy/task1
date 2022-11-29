import numpy as np
import pandas as pd


def generate_cyclical_features(df, col_name, period, start_num=0):
    kwargs = {
        f'sin_{col_name}' : lambda x: np.sin(2*np.pi*(df[col_name]-start_num)/period),
        f'cos_{col_name}' : lambda x: np.cos(2*np.pi*(df[col_name]-start_num)/period)
             }
    return df.assign(**kwargs)

def onehot_encode_pd(df, col_name):
    dummies = pd.get_dummies(df[col_name], prefix=col_name)
    return pd.concat([df, dummies], axis=1).drop(columns=[col_name])

"""
Feature Engineering: Generate features 
week of years
day of week
month
start, middle, end of month
"""
def FeatureGen(df):

    # df =  Dataset.iloc[:, [0,devIdx+1]] 
    df = df.set_index(['time'])

    df.index = pd.to_datetime(df.index)

    # if not df.index.is_monotonic:
    #     df = df.sort_index()

    # df = df.rename(columns={"Dev"+str(devIdx): 'target'})

    #===========================================================
    # df_features
    #===========================================================

    df_features = (
                    df
                    .assign(hour = df.index.hour)
                    .assign(day = df.index.day)
                    .assign(month = df.index.month)
                    .assign(year = df.index.year)
                    .assign(day_of_week = df.index.dayofweek)
                    .assign(week_of_year = df.index.week)
                  )
    
    # weeks = list(np.unique(df_features['week_of_year']))
    weeks = list(np.unique(df_features['week_of_year']))

    df_features = onehot_encode_pd(df_features, 'week_of_year')

    for w in range(53):
        if w+1 not in weeks:
            df_features['week_of_year_'+str(w+1)] = np.zeros(df.shape[0])

    col_names = ['y','hour','day','month','year','day_of_week']
    col_names.extend(['week_of_year_'+str(i+1) for i in range(53)])

    df_features = df_features[col_names]

    df_features = generate_cyclical_features(df_features, 'day_of_week', 7, 0)
    df_features = generate_cyclical_features(df_features, 'month', 12, 1)
    # df_features = generate_cyclical_features(df_features, 'week_of_year', 53, 1)


    df_features = df_features.drop([ 'hour' ], axis = 1)

    # BeginOfMonth
    df_features["BeginOfMonth"] = df_features["day"].transform(lambda x: 1 if x==1 else 0)
    # EndOfMonthValue
    df_features["EndOfMonthValue"] = df_features.groupby(["year","month"])["day"].transform( "max" )
    df_features["day"] = pd.to_numeric(df_features["day"])
    # EndOfMonth [Assumption::: All the month day are complete!!!]
    df_features["EndOfMonth"] = df_features.apply( lambda x: 1 if x["day"]==x["EndOfMonthValue"] else 0, axis=1)
    # MiddleOfMonth
    df_features["MiddleOfMonth"] = df_features.apply( lambda x: 1 if x["day"]>1 and x["day"]<x["EndOfMonthValue"] else 0, axis=1)
    df_features = df_features.drop(["EndOfMonthValue"],axis = 1)

    df_features = df_features.drop(["year"],axis = 1)

    df_features = df_features.drop(["day"],axis = 1)

    for col in df.columns:
        if col not in ['type','y']:
            df_features[col] = df[col]
    # df_features['y'] = df['y']

    return df_features


