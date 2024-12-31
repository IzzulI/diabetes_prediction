import pandas as pd
import numpy as np

def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Renames various ID columns to more readable names.
    """
    # Discharge Disposition
    if 'discharge_disposition_id' in df.columns:
        df.rename(columns={'discharge_disposition_id': 'discharge_disposition'}, inplace=True)
    
    # Admission Type
    if 'admission_type_id' in df.columns:
        df.rename(columns={'admission_type_id': 'admission_type'}, inplace=True)

    # Admission Source
    if 'admission_source_id' in df.columns:
        df.rename(columns={'admission_source_id': 'admission_source'}, inplace=True)

    # Readmitted -> readmission_in_30days
    if 'readmitted' in df.columns:
        df.rename(columns={'readmitted': 'readmission_in_30days'}, inplace=True)

    return df

def remove_patients_who_died(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes rows where 'discharge_disposition' indicates patient died.
    """
    if 'discharge_disposition' in df.columns:
        # Filter out rows with these discharge_disposition codes
        remove_codes = [11, 13, 14, 19, 20, 21]
        df = df[~df['discharge_disposition'].isin(remove_codes)]
    return df

def recode_discharge_disposition(df: pd.DataFrame) -> pd.DataFrame:
    """
    Example recoding:
      - Code '1' -> 'Home'
      - Everything else -> 'Other'
    """
    if 'discharge_disposition' in df.columns:
        home_codes = [1]
        other_codes = list(np.arange(2, 30, 1))

        df['discharge_disposition'].replace(home_codes, 'Home', inplace=True)
        df['discharge_disposition'].replace(other_codes, 'Other', inplace=True)
    return df

def recode_admission_type(df: pd.DataFrame) -> pd.DataFrame:
    """
    Groups admission types into categories:
      - 1 -> 'Emergency'
      - 2 -> 'Emergency'
      - 3 -> 'Elective'
      - 4 -> 'Newborn'
      - 5 -> 'Other'
    """
    if 'admission_type' in df.columns:
        df['admission_type'] = df['admission_type'].astype('object')

        df['admission_type'].replace(2, 1, inplace=True)  
        df['admission_type'].replace(7, 1, inplace=True)
        df['admission_type'].replace(6, 5, inplace=True)
        df['admission_type'].replace(8, 5, inplace=True)

    return df

def recode_admission_source(df: pd.DataFrame) -> pd.DataFrame:
    """
    Groups admission sources into categories:
      - 1,2,3 -> 'Physician Referral'
      - 7,16  -> 'Emergency Room'
      - Everything else -> 'Other'
    """
    if 'admission_source' in df.columns:
        phy_ref = [1, 2, 3]
        emer_room = [7, 16]
        # We'll treat everything else as 'Other' for brevity

        df['admission_source'].replace(phy_ref, 'Physician Referral', inplace=True)
        df['admission_source'].replace(emer_room, 'Emergency Room', inplace=True)

        # Convert anything still numeric to 'Other'
        mask_numeric = df['admission_source'].apply(lambda x: isinstance(x, (int, float)))
        df.loc[mask_numeric, 'admission_source'] = 'Other'
    return df

def recode_readmission(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms readmission_in_30days from 'NO','>30','<30' to binary '0'/'1'.
    Also renames column to reflect this new meaning.
    """
    if 'readmission_in_30days' in df.columns:
        df['readmission_in_30days'].replace('NO', '0', inplace=True)
        df['readmission_in_30days'].replace('>30', '0', inplace=True)
        df['readmission_in_30days'].replace('<30', '1', inplace=True)
        # Ensure the column is typed as integer or category
        df['readmission_in_30days'] = df['readmission_in_30days'].astype('category')
    return df

def categorize_diag_1(df: pd.DataFrame) -> pd.DataFrame:
    """
    Example for grouping ICD codes in 'diag_1' into broader categories.
    - Fill NaN with '' 
    - Keep first 3 digits
    - Map to a textual category like 'circulatory','respiratory','injury', etc
    """
    if 'diag_1' in df.columns:
        # Fill missing
        df['diag_1'] = df['diag_1'].fillna('others')
        # Keep only the first digit or first 3 digits
        df['diag_1'] = df['diag_1'].astype(str).str.split('.').str.get(0)

        def map_icd(icd):
            # Return 'others' for blank or if not numeric
            if icd == '' or not icd.isdigit():
                return 'others'
            
            # Convert to int safely
            code = int(icd)

            if code == 250:
                return 'diabetes mellitus'
            elif code == 782:
                return 'skin'
            elif code == 785:
                return 'circulatory'
            elif code == 786:
                return 'respiratory'
            elif code == 787:
                return 'digestive'
            elif code == 788:
                return 'genitourinary'
            elif code <= 139:
                return 'others'  # Infectious/Parasitic
            elif code <= 239:
                return 'neoplasms'
            elif code <= 279:
                return 'others'  # Endocrine/Nutritional
            elif code <= 289:
                return 'others'  # Blood etc.
            elif code <= 319:
                return 'others'  # Mental disorders
            elif code <= 389:
                return 'others'  # Nervous system
            elif code <= 459:
                return 'circulatory'
            elif code <= 519:
                return 'respiratory'
            elif code <= 579:
                return 'digestive'
            elif code <= 629:
                return 'genitourinary'
            elif code <= 679:
                return 'others'  # pregnancy/childbirth
            elif code <= 709:
                return 'skin'
            elif code <= 739:
                return 'musculoskeletal'
            elif code <= 759:
                return 'others'  # congenital anomalies
            elif code <= 779:
                return 'others'  # perinatal conditions
            elif code <= 799:
                return 'others'  # symptoms/ill-defined
            elif code <= 999:
                return 'injury'
            else:
                return 'others'
        
        df['diag_1'] = df['diag_1'].apply(map_icd)

    return df

def encode_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encodes categorical columns according to a predefined 'mappings' dictionary,
    where each column maps original string (or category) values to integer codes.
    """

    mappings = {
        'race': {
            value: idx for idx, value in enumerate(
                ['Caucasian', 'AfricanAmerican', 'Other', 'Asian', 'Hispanic']
            )
        },
        'gender': {
            value: idx for idx, value in enumerate(['Female', 'Male'])
        },
        'age': {
            value: idx for idx, value in enumerate([
                '[0-10)', '[10-20)', '[20-30)', '[30-40)',
                '[40-50)', '[50-60)', '[60-70)', '[70-80)',
                '[80-90)', '[90-100)'
            ])
        },
        'admission_type': {
            value: idx for idx, value in enumerate([5, 1, 3, 4])
        },
        'discharge_disposition': {
            value: idx for idx, value in enumerate(['Other', 'Home'])
        },
        'admission_source': {
            value: idx for idx, value in enumerate([
                'Physician Referral', 'Emergency Room', 'Other'
            ])
        },
        'diag_1': {
            value: idx for idx, value in enumerate([
                'diabetes mellitus', 'others', 'neoplasms',
                'circulatory', 'respiratory', 'injury',
                'musculoskeletal', 'digestive', 'genitourinary'
            ])
        },
        'max_glu_serum': {
            value: idx for idx, value in enumerate(['None', '>300', 'Norm', '>200'])
        },
        'A1Cresult': {
             value: idx for idx, value in enumerate(['None', '>7', '>8', 'Norm'])
        },
        'metformin': {
            value: idx for idx, value in enumerate(['No', 'Steady', 'Up', 'Down'])
        },
        'glimepiride': {
            value: idx for idx, value in enumerate(['No', 'Steady', 'Down', 'Up'])
        },
        'glipizide': {
            value: idx for idx, value in enumerate(['No', 'Steady', 'Up', 'Down'])
        },
        'glyburide': {
            value: idx for idx, value in enumerate(['No', 'Steady', 'Up', 'Down'])
        },
        'pioglitazone': {
            value: idx for idx, value in enumerate(['No', 'Steady', 'Up', 'Down'])
        },
        'rosiglitazone': {
            value: idx for idx, value in enumerate(['No', 'Steady', 'Up', 'Down'])
        },
        'insulin': {
            value: idx for idx, value in enumerate(['No', 'Up', 'Steady', 'Down'])
        },
        'change': {
            value: idx for idx, value in enumerate(['No', 'Ch'])
        },
        'diabetesMed': {
            value: idx for idx, value in enumerate(['No', 'Yes'])
        },
        'readmission_in_30days': {
            value: idx for idx, value in enumerate(['0', '1'])
        }
    }

    df_encoded = df.copy()

    # Apply the mapping to each column that appears in 'mappings'
    for column, mapping in mappings.items():
        # Only map if the column actually exists in df
        if column in df_encoded.columns:
            df_encoded[column] = df_encoded[column].map(mapping)
    
    df_encoded['diag_1'] = df_encoded['diag_1'].fillna(0)
    df_encoded['max_glu_serum'] = df_encoded['max_glu_serum'].fillna(0)
    df_encoded['A1Cresult'] = df_encoded['A1Cresult'].fillna(0)
    return df_encoded

def transform_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Master function to call all transformation steps in sequence.
    """
    # Rename ID columns to readable names
    df = rename_columns(df)

    # Remove rows for patients who died (if you don’t want them in analysis)
    df = remove_patients_who_died(df)

    # Recode discharge disposition
    df = recode_discharge_disposition(df)

    # Recode admission type
    df = recode_admission_type(df)

    # Recode admission source
    df = recode_admission_source(df)

    # Turn readmission into binary
    df = recode_readmission(df)

    # Categorize diag_1
    df = categorize_diag_1(df)

    # Encode some categorical columns
    df = encode_categorical_features(df)

    return df

if __name__ == "__main__":
    path = "data/cleaned_diabetic_data.csv"
    df_test = pd.read_csv(path)
    
    # Transform
    df_transformed = transform_data(df_test)
    
    df_transformed.to_csv("data/transformed_diabetic_data.csv", index=False)
    
    print("Data shape after transformation:", df_transformed.shape)
    print("\nSample rows:\n", df_transformed.head())