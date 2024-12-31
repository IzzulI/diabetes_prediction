import pandas as pd

# Load preprocessed data into dataset
df_ready2 = pd.read_csv('data/transformed_diabetic_data.csv')

# Split data into input and labels
X = df_ready2.drop('readmission_in_30days', axis=1)
y = df_ready2['readmission_in_30days']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)

# Save to csv
X_train.to_csv('data/X_train.csv', index=False)
y_train.to_csv('data/y_train.csv', index=False)
X_test.to_csv('data/X_test.csv', index=False)
y_test.to_csv('data/y_test.csv', index=False)
