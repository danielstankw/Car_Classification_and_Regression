

def preprocessing_target_variables(df):
    """
    Process a given DataFrame to clean and group specific target variables related to vehicle trims.

    Parameters:
    - df (pd.DataFrame): A DataFrame containing vehicle data. 
                         Requires columns 'Vehicle_Trim' and 'Dealer_Listing_Price'.

    Returns:
    - pd.DataFrame: A DataFrame after performing preprocessing steps on 'Vehicle_Trim' values and removing rows
                    with specific trims.

    Notes:
    - The function first handles missing data, grouping specific trims for both Jeep and Cadillac brands, 
      and removing specific trims from the DataFrame.
    - Trims are grouped according to mappings defined within the function for Jeep and Cadillac.
    - Specific trims are dropped from the dataset as specified by the function logic.
    """
    
    df_copy = df.copy()
    df_copy = df_copy.dropna(subset=['Vehicle_Trim', 'Dealer_Listing_Price'])

    # Mapping dictionary
    jeep_mapping = {
        'Limited 75th Anniversary Edition': 'Limited',
        'Limited X': 'Limited',
        'Limited 4x4': 'Limited',
        'Limited 75th Anniversary': 'Limited',

        'Laredo E': 'Laredo',

        'SRT Night': 'SRT',

        'Summit': 'Overland',

        'Trackhawk': 'Trailhawk',

        'High Altitude': 'Altitude'}

    # Replace and groupby
    df_copy['Vehicle_Trim'] = df_copy['Vehicle_Trim'].replace(jeep_mapping)
    # dropping 
    jeep_drop_trims = ['75th Anniversary Edition', 'Upland','75th Anniversary','Sterling Edition']

    df_copy = df_copy[~df_copy['Vehicle_Trim'].isin(jeep_drop_trims)]

    # Mapping dictionary
    cadillac_mapping = {
        'Premium Luxury AWD': 'Premium',
        'Premium Luxury FWD': 'Premium',
        'Premium Luxury': 'Premium',

        'Luxury FWD': 'Luxury',
        'Luxury': 'Luxury',
        'Luxury AWD': 'Luxury',

        'Base': 'Base',
        'FWD': 'Base',

        'Platinum': 'Platinum',
        'Platinum AWD': 'Platinum'}

    # Replace and groupby
    df_copy['Vehicle_Trim'] = df_copy['Vehicle_Trim'].replace(cadillac_mapping)
    
    return df_copy