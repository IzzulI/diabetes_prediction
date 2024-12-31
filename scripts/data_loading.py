import pandas as pd

def load_data(path_to_file: str) -> pd.DataFrame:
    df = pd.read_csv(path_to_file)
    return df

if __name__ == "__main__":
    data = load_data("data/diabetic_data.csv")
    print(data.head())