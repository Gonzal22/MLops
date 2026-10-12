import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(filepath="heart.csv", test_size=0.2, random_state=42):
    """Carga, preprocesa y divide los datos."""
    df = pd.read_csv(filepath)
    
    # Tratamiento de nulos
    df['Cholesterol'] = df['Cholesterol'].replace(0, np.nan)
    df['RestingBP'] = df['RestingBP'].replace(0, np.nan)
    df['Cholesterol'] = df['Cholesterol'].fillna(df.groupby('HeartDisease')['Cholesterol'].transform('median'))
    df['RestingBP'] = df['RestingBP'].fillna(df.groupby('HeartDisease')['RestingBP'].transform('median'))
    
    # Winsorización
    variables_numericas = ['RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
    for col in variables_numericas:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        limite_inferior = Q1 - 1.5 * IQR
        limite_superior = Q3 + 1.5 * IQR
        df[col] = np.clip(df[col], limite_inferior, limite_superior)
        
    # One-Hot Encoding
    cat_col = ['Sex', 'ChestPainType', 'FastingBS', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
    df_clean = pd.get_dummies(df, columns=cat_col, drop_first=True)
    
    # Separar X e y
    X = df_clean.drop('HeartDisease', axis=1)
    y = df_clean['HeartDisease']
    
    # Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    return X_train, X_test, y_train, y_test