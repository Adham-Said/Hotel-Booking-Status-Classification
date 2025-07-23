import pandas as pd
from sklearn.preprocessing import OneHotEncoder, PowerTransformer, RobustScaler

df6 = pd.read_csv('first inten project.csv')


numerical_features = [['lead time', 'average price ', 'P-C', 'P-not-C', 'num_of_individuals',
                                'number of week nights', 'number of weekend nights', 'special requests']]
categorical_features = [['type of meal', 'car parking space', 'room type', 'market segment type', 'repeated']]

def feature_engineering(dataframe):
    df = dataframe.copy()
    df['num_of_individuals'] = df['number of adults'] + df['number of children']
    df.drop(['number of adults', 'number of children'], axis = 1, inplace = True)
    return df

def parse_dates(dataframe, date_column):
    df = dataframe.copy()
    df[date_column] = pd.to_datetime(df[date_column], errors = 'coerce')
    df = df.sort_values(by = date_column)
    df['month'] = df[date_column].dt.month
    df['year'] =df[date_column].dt.year
    df['day'] = df[date_column].dt.day
    df.drop([date_column, 'Booking_ID'], axis = 1, inplace = True)
    df.dropna(inplace = True)
    return df


def apply_encoding(dataframe, encoder):
    df = dataframe.copy()
    df['type of meal'] = df['type of meal'].str.replace("Meal Plan ", "")
    df['type of meal'] = df['type of meal'].str.replace("Not Selected", "-1")
    df['type of meal'] = df['type of meal'].astype(int)
    df['room type'] = df['room type'].str.replace("Room_Type ", "")
    df['room type'] = df['room type'].astype(int)
    ohe_array = encoder.transform(dataframe[['market segment type']])
    ohe_columns = pd.DataFrame(ohe_array,
                                columns = encoder.get_feature_names_out(dataframe[['market segment type']].columns)
                               , index = dataframe.index)
    df = pd.concat([df.drop(columns = 'market segment type'), ohe_columns], axis = 1)
    return df


robust_scaler = RobustScaler()
log_transformer = PowerTransformer()
def apply_scaler(dataframe, columns):
    df = dataframe.copy()
    for column in columns:
        if column == 'lead time':
            df['lead time'] = log_transformer.transform(df['lead time'])
        # df[column] = robust_scaler.transform(df[column])

    return df


# encoder = OneHotEncoder(sparse_output=False, handle_unknown = 'ignore')
# df7 = feature_engineering(df6)
# df8 = parse_dates(df7, 'date of reservation')
# df9 = apply_encoding(df8, encoder)
# df10 = apply_scaler(df9, numerical_features)
# # print(df7['type of meal'].unique())
# # print(df7['room type'].unique())
# print(df10.isnull().sum())
                                  
                                  
    


