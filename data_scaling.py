import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split


# 현재 path = "spotify_churn_dataset.csv" 
def load_data(path):
    df = pd.read_csv(path, skipinitialspace=True)
    return df

def preprocessor():
    categorical_columns = ['gender','country','subscription_type','device_type'] 
    numeric_columns = ['age','listening_time','songs_played_per_day', 'skip_rate','ads_listened_per_week','offline_listening'] # user_id 는 의미 없어서 지움

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())  
    ])

    category_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")), 
        ("ohe", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("category", category_pipeline, categorical_columns), 
        ("number", numeric_pipeline, numeric_columns)
    ])

    return preprocessor

def preprocess_data(df):
    x = df.drop(['user_id', 'is_churned'],axis=1) # is_churned 가 열에 있어서 axis = 1
    y = df['is_churned']
    
    x_train, x_test, y_train, y_test = train_test_split(x , y, test_size=0.2, random_state=0)

    processor = preprocessor()

    x_train = processor.fit_transform(x_train)
    x_test = processor.transform(x_test)

    return x_train, x_test, y_train, y_test