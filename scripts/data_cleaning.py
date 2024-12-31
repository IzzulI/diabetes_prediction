import pandas as pd

def replace_missing_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replaces known 'invalid' or placeholder strings (like '?' or 'Unknown/Invalid')
    with pandas' NA (Not Available).
    """
    df.replace('?', pd.NA, inplace=True)
    df.replace('Unknown/Invalid', pd.NA, inplace=True)
    return df

def drop_irrelevant_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drops columns that are not useful or are mostly missing:
      - 'weight'
      - 'payer_code'
      - 'medical_specialty'
      - 'diag_2' and 'diag_3' (leaving diag_1 as the primary diagnosis)
    """
    cols_to_drop = ['weight', 'payer_code', 'medical_specialty', 'diag_2', 'diag_3']
    df.drop(columns=cols_to_drop, inplace=True, errors='ignore')
    return df

def drop_missing_gender(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drops rows where 'gender' is missing, since it's considered essential.
    """
    df.dropna(subset=['gender'], inplace=True)
    return df

def replace_na_in_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replaces NA values in certain columns with specific labels:
      - 'race' -> 'Other'
      - 'A1Cresult' -> 'None' (meaning 'not measured')
      - 'max_glu_serum' -> 'None' (meaning 'not measured')
    """
    # Replace race NA with 'Other'
    df['race'].replace(pd.NA, 'Other', inplace=True)
    # Replace A1Cresult NA with 'None'
    df['A1Cresult'].replace(pd.NA, 'None', inplace=True)
    # Replace max_glu_serum NA with 'None'
    df['max_glu_serum'].replace(pd.NA, 'None', inplace=True)

    return df

def remove_duplicate_patients(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keeps only the first (primary) observation for each patient.
    Then drops 'patient_nbr' from the DataFrame.
    Also drops 'encounter_id' since it's not needed after deduplication.
    """
    # Deduplicate by 'patient_nbr' (keep first observation)
    df.drop_duplicates(subset='patient_nbr', keep='first', inplace=True)
    
    # Drop columns we no longer need
    cols_to_drop = ['patient_nbr', 'encounter_id']
    df.drop(columns=cols_to_drop, inplace=True, errors='ignore')
    
    return df

def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes rows where certain columns exceed thresholds based on domain research:
      - number_inpatient   > 12
      - number_emergency   > 20
      - number_outpatient  > 14
    """
    df = df[df['number_inpatient'] <= 12]
    df = df[df['number_emergency'] <= 20]
    df = df[df['number_outpatient'] <= 14]
    return df

def drop_imbalanced_features(df: pd.DataFrame, threshold: float = 0.95) -> pd.DataFrame:
    """
    Drops categorical features that are too imbalanced, 
    i.e., if a single category accounts for > 'threshold' ratio of rows.
    """
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    to_drop = []
    for col in categorical_cols:
        # Check the fraction of the most frequent category
        top_freq = df[col].value_counts(normalize=True).max()
        if top_freq > threshold:
            to_drop.append(col)
    if to_drop:
        df.drop(columns=to_drop, inplace=True, errors='ignore')
        print(f"Dropped imbalanced columns (threshold={threshold}): {to_drop}")
    else:
        print("No columns dropped due to imbalance.")
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Master function to apply all cleaning steps in sequence.
    Adjust the order as needed based on your project's requirements.
    """
    # Replace known placeholders with NA
    df = replace_missing_indicators(df)
    
    # Drop columns with high missing or irrelevance
    df = drop_irrelevant_columns(df)
    
    # Drop rows where gender is missing (essential info)
    df = drop_missing_gender(df)
     
    # Replace missing in certain columns with domain-specific labels
    df = replace_na_in_columns(df)
    
    # Remove duplicate patient records
    df = remove_duplicate_patients(df)
    
    # Remove outliers based on domain thresholds
    df = remove_outliers(df)

    # Drop imbalanced features with a threshold of 98%
    df = drop_imbalanced_features(df, threshold=0.98)
    
    return df

if __name__ == "__main__":
    test_path = "data/diabetic_data.csv"
    df_test = pd.read_csv(test_path)
    df_cleaned = clean_data(df_test)
    df_cleaned.to_csv("data/cleaned_diabetic_data.csv", index=False)
    print("\nData shape after cleaning:", df_cleaned.shape)
    print("\nSample rows:\n", df_cleaned.head())
    