import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(file_path):
    """Load dataset from a CSV file."""
    return pd.read_csv(file_path)

def split_data(df, label_column, test_size=0.2, random_state=123, stratify_column=None):
    """Split the dataset into input features and labels, then into train and test sets."""
    X = df.drop(label_column, axis=1)
    y = df[label_column]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify_column
    )
    return X_train, X_test, y_train, y_test

def save_data(data, file_path):
    """Save a DataFrame or Series to a CSV file."""
    data.to_csv(file_path, index=False)

def split(data_path='data/selected_diabetic_data.csv', label_column='readmission_in_30days', output_dir='data/', seed=123):
    """
    Split the dataset into training and test sets, and save them to CSV files.
    
    Parameters:
        data_path (str): Path to the input CSV file.
        label_column (str): Name of the label column in the dataset.
        output_dir (str): Directory to save the output files.
        seed (int): Random seed for reproducibility.
    """
    try:
        # Load data
        df_ready2 = load_data(data_path)
        
        # Split data
        X_train, X_test, y_train, y_test = split_data(
            df_ready2, label_column, test_size=0.2, random_state=seed, stratify_column=df_ready2[label_column]
        )
        
        # Save to CSV files
        save_data(X_train, f'{output_dir}X_train.csv')
        save_data(y_train, f'{output_dir}y_train.csv')
        save_data(X_test, f'{output_dir}X_test.csv')
        save_data(y_test, f'{output_dir}y_test.csv')

        print("Data split and saved successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    split()