import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
import pandas as pd
from src.entity import DataPreprocessingConfig
from logger_config import get_logger

logger = get_logger(__name__)

class DataPreprocessing:
    def __init__(self, config: DataPreprocessingConfig):
        self.config = config

    def date_to_epoch_time(self, df, columns):
        for col in columns:
            try:
                # Convert the column to datetime format
                df[col] = pd.to_datetime(df[col], errors='raise')
                
                # Convert datetime to epoch time (in seconds)
                df[col] = df[col].astype(np.int64) / 10**9
                logger.info(f"Converted '{col}' to epoch time.")
            except Exception as e:
                logger.info(f"Skipping column '{col}' due to: {e}")

        return df
      
    def object_encoder(self, df, binary_features, ordinal_features, nominal_features):        
        le = LabelEncoder()
        for feature in binary_features:
            df[feature] = le.fit_transform(df[feature])
        
        for feature, order in ordinal_features.items():
            oe = OrdinalEncoder(categories=[order])
            df[feature] = oe.fit_transform(df[[feature]]).flatten()
        
        dummies = pd.get_dummies(df[nominal_features], drop_first=True).astype('bool')
        df = pd.concat([df.drop(columns=nominal_features), dummies], axis=1)

        return df

    def data_imputer(self, df):
        numeric_columns = df.select_dtypes(include=['number']).columns
        for col in numeric_columns:
            if col in df.columns:
                df[col].fillna(-1, inplace=True)

        object_columns = df.select_dtypes(include=['object']).columns
        for col in object_columns:
            if col in df.columns:
                df[col].fillna("Unknown", inplace=True)
        return df
        
    def data_preprocessing_pipeline(self):
        train_data_path = os.path.join(self.config.source, 'train.csv')
        test_data_path = os.path.join(self.config.source, 'test.csv')
        train_df = pd.read_csv(train_data_path, index_col = [0])
        test_df = pd.read_csv(test_data_path, index_col = [0])
        logger.info("imputing Training and Testing data")
        train_df = self.data_imputer(train_df)
        test_df = self.data_imputer(test_df)
        logger.info("Training and Testing data imputation done") 
        
        logger.info("Converting date-time column to epoch time")
        train_columns = train_df.select_dtypes(include=['object']).columns
        test_columns = test_df.select_dtypes(include=['object']).columns
        train_df = self.date_to_epoch_time(train_df, train_columns)
        test_df = self.date_to_epoch_time(test_df, test_columns)
        logger.info("Date-time to epoch converted")

        logger.info("Encoding object columns")
        binary_features = ['Gender', 'Smoking Status']
        ordinal_features = {
            'Exercise Frequency': ['Rarely', 'Monthly', 'Weekly', 'Daily']
        }
        nominal_features = ['Marital Status', 'Education Level', 'Occupation', 
                            'Location', 'Policy Type', 'Customer Feedback', 'Property Type']
        train_df = self.object_encoder(train_df, binary_features, ordinal_features, nominal_features)
        test_df = self.object_encoder(test_df, binary_features, ordinal_features, nominal_features)
        logger.info("object columns encoded")
        
        logger.info("Scaling Training and Testing Data")
        numerical_columns= train_df.select_dtypes('float64').columns
        numerical_columns = numerical_columns[numerical_columns != "Premium Amount"]
        scaler = StandardScaler()
        scaler.fit(train_df[numerical_columns])
        train_df[numerical_columns] = scaler.fit_transform(train_df[numerical_columns])
        test_df[numerical_columns] = scaler.fit_transform(test_df[numerical_columns])
        logger.info("Training and Testing Data scaled")

        logger.info("Log-transforming 'Annual Income and Premium Amount' and improving column name")
        train_df['Annual Income'] = np.log1p(train_df['Annual Income'])
        test_df['Annual Income'] = np.log1p(test_df['Annual Income'])
        train_df['Premium Amount'] = np.log1p(train_df['Premium Amount'])

        train_df.columns = train_df.columns.str.replace(' ', '_', regex=True)
        test_df.columns = test_df.columns.str.replace(' ', '_', regex=True)
        logger.info("Previous process done")

        logger.info("Making directory for storing preprocessed data")
        root_dir = self.config.root_dir
        os.makedirs(root_dir, exist_ok = True)
        logger.info(f"{root_dir} made")
        train_file_path = os.path.join(root_dir, 'train_cleaned.csv')
        test_file_path = os.path.join(root_dir, 'test_cleaned.csv')
        train_df.to_csv(train_file_path, index=False)
        test_df.to_csv(test_file_path, index=False)
        print(f"Preprocessed Training and Testing data saved in {root_dir}")
