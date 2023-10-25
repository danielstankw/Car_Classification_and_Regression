from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
import re

class SetIndex(BaseEstimator, TransformerMixin):
    """
    Transforms a pd.DataFrame by setting a feature as its index.
    
    Parameters:
    ----------
    index_feature : str
        The feature to be set as the DataFrame index.
    """
    def __init__(self, index_feature):
        self.index_feature = index_feature
    
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_copy = X.copy()
        X_copy.set_index(self.index_feature, inplace=True)
        return X_copy
    
class DropUnused(BaseEstimator, TransformerMixin):
    """
    Transformer that drops specified features from a pd.DataFrame.
    
    Parameters:
    ----------
    features : list
        List of features to be dropped.
    """
    def __init__(self, features):
        self.features = features

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_copy = X.copy()
        return X_copy.drop(self.features, axis=1)
    
class LogTransform(BaseEstimator, TransformerMixin):
    """
    Applies a log transformation on specified features of a pd.DataFrame.
    Handles NaN values by filling them with the mean of the transformed feature.
    
    Parameters:
    ----------
    features : list
        List of features to apply the log transformation.
    """
    def __init__(self, features):
        self.features = features
        self.feature_means_ = {}

    def fit(self, X, y=None):
        X_copy = X.copy()
        
        for feature in self.features:
            X_copy[feature] = np.log1p(X_copy[feature])
            self.feature_means_[feature] = X_copy[feature].mean()
        return self

    def transform(self, X, y=None):
        X_copy = X.copy()
        for feature in self.features:
            X_copy[feature] = np.log1p(X_copy[feature])
            X_copy[feature].fillna(self.feature_means_[feature], inplace=True)
        return X_copy    
    
class BooleanEncoding(BaseEstimator, TransformerMixin):
    """
    Converts specified features from boolean to integer representation.
    
    Parameters:
    ----------
    features : list
        List of features to be converted to int.
    """
    def __init__(self, features):
        self.features = features

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_copy = X.copy()
        for feature in self.features:
            X_copy[feature] = X_copy[feature].astype(int)
        return X_copy
    
class ColumnNameTransformer(BaseEstimator, TransformerMixin):
    """
    Transforms column names of a pd.DataFrame by replacing spaces with underscores.
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_copy = X.copy()
        X_copy.columns = [col.replace(' ', '_') for col in X_copy.columns]
        return X_copy
    
    
class AgeFeatureTransform(BaseEstimator, TransformerMixin):
    """
    Calculates vehicle age from a given year and the vehicle's year of manufacture.
    
    Parameters:
    ----------
    current_year : int
        The current year used to calculate the vehicle's age.
    """
    def __init__(self, current_year):
        self.current_year = current_year

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_copy = X.copy()
        X_copy['VehAge'] = self.current_year - X_copy['VehYear']
        X_copy.drop('VehYear', axis=1, inplace=True)
        return X_copy
    
    
class VehHistoryTransform(BaseEstimator, TransformerMixin):
    """
    Extracts and transforms various vehicle history information from a text feature.
    Imputes missing values with most frequent value present in the column.
    """
    def __init__(self):
        self.imputation_dict_ = {}

    def fit(self, X, y=None):

        X_temp = self._transform_logic(X.copy())  
        
        self.imputation_dict_['Num_Owners'] = X_temp['Num_Owners'].mode()[0]
        self.imputation_dict_['AccidentReported'] = X_temp['AccidentReported'].mode()[0]
        self.imputation_dict_['NonPersonalUse'] = X_temp['NonPersonalUse'].mode()[0]
        self.imputation_dict_['TitleIssues'] = X_temp['TitleIssues'].mode()[0]
        self.imputation_dict_['BuybackProtection'] = X_temp['BuybackProtection'].mode()[0]

        return self

    def transform(self, X, y=None):
        X_copy = self._transform_logic(X.copy())
        for feature in self.imputation_dict_:
            # for each feature we impute NaNs with most frequent and than conver to int
            X_copy[feature].fillna(self.imputation_dict_[feature], inplace=True)
            X_copy[feature] = X_copy[feature].astype(int)        
         
        return X_copy

    def _transform_logic(self, df):
        # Extract number of owners and create a new column
        df['Num_Owners'] = df['VehHistory'].str.extract(r'(\d+) Owner').astype(float)
        # Check if accidents were reported
        df['AccidentReported'] = df['VehHistory'].str.contains(r'Accident\(s\) Reported').astype(bool)
         # Check if non-personal use was reported
        df['NonPersonalUse'] = df['VehHistory'].str.contains('Non-Personal Use Reported').astype(bool)
         # Check for title issues
        df['TitleIssues'] = df['VehHistory'].str.contains(r'Title Issue\(s\) Reported').astype(bool)
        # Check for buyback protection eligibility
        df['BuybackProtection'] = df['VehHistory'].str.contains('Buyback Protection Eligible').astype(bool)

        df.drop('VehHistory', axis=1, inplace=True)
        
        return df
    
    
class VehEngineTransform(BaseEstimator, TransformerMixin):
    """
    Extracts and categorizes information about the vehicle's engine by creating two 
    new features: Displacement & Engine_Category.
    """
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        X_transformed = X.copy()

        X_transformed['Displacement'] = X_transformed['VehEngine'].apply(self._get_displacement)
        X_transformed['Engine_Category'] = X_transformed['VehEngine'].apply(self._categorize_engine)
        
        X_transformed.drop('VehEngine', axis=1, inplace=True)
        
        return X_transformed
    
    def _get_displacement(self, engine):
        if pd.isna(engine) or not isinstance(engine, str):
            return 0
        match = re.search(r'(\d+\.\d+)', engine)
        
        return float(match.group(1)) if match else 0

    def _categorize_engine(self, engine):
        engine_types = ['turbo', 'supercharged', 'diesel', 'hemi']

        if pd.isna(engine) or not isinstance(engine, str):
            return 'Other'
        for etype in engine_types:
            if etype in engine.lower():
                return etype.capitalize()
        return 'Other'
        
        
        
class VehColorExtTransform(BaseEstimator, TransformerMixin):
    """
    Categorizes external vehicle colors into basic colors.
    """
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        X_transformed = X.copy()
        
        # impute NaN with White
        def basic_color(color):
            if pd.isna(color):
                return 'White'
            elif not isinstance(color, str):
                return 'Other'

            base_colors = ['Black', 'White', 'Red', 'Silver', 'Blue', 'Gray', 'Brown', 'Gold']
            for base in base_colors:
                if base.lower() in color.lower():
                    return base
            return 'Other'

        X_transformed['BasicExtColor'] = X_transformed['VehColorExt'].apply(basic_color)
        X_transformed.drop('VehColorExt', axis=1, inplace=True)
        
        return X_transformed

    
    
class VehDriveTrainTransform(BaseEstimator, TransformerMixin):
    """
    Transforms and categorizes drive train information of a vehicle into 
    3 categories: front_drive, all_drive and other.
    """
    
    drivetrain_mapping = {
        '4x4/4WD': 'all_drive','4WD': 'all_drive','FWD': 'front_drive',
        'AWD': 'all_drive','4x4': 'all_drive','Four Wheel Drive': 'all_drive',
        '4X4': 'all_drive','All Wheel Drive': 'all_drive',
        'ALL-WHEEL DRIVE WITH LOCKING AND LIMITED-SLIP DIFFERENTIAL': 'all_drive',
        'AWD or 4x4': 'all_drive','Front Wheel Drive': 'all_drive',
        'All-wheel Drive': 'all_drive','ALL WHEEL': 'all_drive',
        'AllWheelDrive': 'all_drive','4WD/AWD': 'all_drive'
    }

    def __init__(self):
        self.most_frequent_drive_ = None
    
    def fit(self, X, y=None):
        X_transformed = X['VehDriveTrain'].replace(self.drivetrain_mapping)
        # Determine the most frequent value after mapping
        self.most_frequent_drive_ = X_transformed.mode()[0]
        return self
    
    def transform(self, X, y=None):
        X_transformed = X.copy()
        
        X_transformed['VehDrive'] = X_transformed['VehDriveTrain'].replace(self.drivetrain_mapping)
        X_transformed['VehDrive'].fillna(self.most_frequent_drive_, inplace=True)
               
        # Handle unknown categories
        known_values = set(self.drivetrain_mapping.values())
        X_transformed.loc[~X_transformed['VehDrive'].isin(known_values), 'VehDrive'] = 'other'
        
        X_transformed.drop('VehDriveTrain', axis=1, inplace=True)
        return X_transformed
    
    
    
class VehMileageTransform(BaseEstimator, TransformerMixin):
    """
    Handles missing values in vehicle mileage by imputing them with the mean value.
    """
    def __init__(self):
        self.mileage_mean_ = None
    
    def fit(self, X, y=None):
        self.mileage_mean_ = X['VehMileage'].mean()
        return self
    
    def transform(self, X, y=None):
        X_transformed = X.copy()
        
        # impute NaN with mean values computed during fit
        X_transformed['VehMileage'].fillna(self.mileage_mean_, inplace=True)
        return X_transformed

    
    
class VehMakeTransform(BaseEstimator, TransformerMixin):
    """
    Transforms vehicle make information to identify specific brands.
    """
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        X_transformed = X.copy()
        
        vehmake_mapping = {"Jeep": 1, "Cadillac": 0}
        X_transformed['Is_Jeep'] = X_transformed['VehMake'].map(vehmake_mapping)
        X_transformed.drop('VehMake', axis=1, inplace=True)
        return X_transformed
    
    
    
class VehColorInternalTransform(BaseEstimator, TransformerMixin):
    """
    Extracts information about the presence of leather in the internal color description.
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # in case of NaN --> False
        X_transformed = X.copy()
        
        X_transformed['ContainsLeather'] = X_transformed['VehColorInt'].str.contains('leather', case=False, na=False).astype(bool)
        X_transformed.drop('VehColorInt', axis=1, inplace=True)
        X_transformed['ContainsLeather'] = X_transformed['ContainsLeather'].astype(int)
        return X_transformed
    
    
    
class VehFeatsTransform(BaseEstimator, TransformerMixin):
    """
    Transforms the vehicle features information by counting the number of features.
    """
    def __init__(self):
        # store most frequent value from train data for imputation
        self.most_freq_ = None
    
    def fit(self, X, y=None):
        # if Nan -> None
        # determine most freq value in the train data
        veh_feats_count = X['VehFeats'].apply(lambda x: len(str(x).split(',')) if not pd.isna(x) else None)
        self.most_frequent_ = veh_feats_count.mode()[0]
        return self
    
    def transform(self, X, y=None):
        # avoid inplace modification
        X_transformed = X.copy()
        
        X_transformed['VehFeatsCount'] = X_transformed['VehFeats'].apply(lambda x: len(str(x).split(',')) if not pd.isna(x) else None)
        # Use the most frequent value for imputation of NaN values
        X_transformed['VehFeatsCount'].fillna(self.most_frequent_, inplace=True)
        X_transformed.drop('VehFeats', axis=1, inplace=True)
        return X_transformed
    
