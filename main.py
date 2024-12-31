from datetime import datetime
from time import time
from scripts.data_loading import load_data
from scripts.data_cleaning import clean_data
from scripts.data_transformation import transform_data
from scripts.univariate_analysis import univariate_analysis
from scripts.heatmap import heatmap, feature_selection
from scripts.bivariate_analysis import bivariate_analysis
from scripts.train_test_split import split
from scripts.catboost_bayesian import get_catboost_bayesian
from scripts.catboost_grid import get_catboost_grid
from scripts.catboost_random import get_catboost_random
from scripts.lightgbm_bayesian import get_lightgbm_bayesian
from scripts.lightgbm_grid import get_lightgbm_grid
from scripts.lightgbm_random import get_lightgbm_random
from scripts.xgboost_bayesian import get_xgboost_bayesian
from scripts.xgboost_grid import get_xgboost_grid
from scripts.xgboost_random import get_xgboost_random
from scripts.evaluation import get_evaluation

def main():
    # Store prepocessing start time
    start_preprocessing=time()

    # Load Data
    df = load_data("data/diabetic_data.csv")

    # Clean Data
    df = clean_data(df)

    # Transform Data
    df = transform_data(df)

    df.to_csv("data/transformed_diabetic_data.csv", index=False)

    # Store EDA start time
    start_eda=time()

    # # Univariate Analysis
    univariate_analysis(df)

    # # Heatmap
    heatmap(df)

    # # Bivariate Analysis
    bivariate_analysis(df)

    # Feature Selection
    df = feature_selection(df)

    df.to_csv("data/selected_diabetic_data.csv", index=False)    

    #Store modeling starting time
    start_modeling = time()

    # Split Data
    split(data_path='data/selected_diabetic_data.csv', label_column='readmission_in_30days', output_dir='data/', seed=123)

    # Modeling with Hyperparameter Tuning (Can be run in parallel)
    get_catboost_bayesian('data/X_train.csv', 'data/y_train.csv', 'models/catboost_bayes.pkl')
    get_catboost_grid('data/X_train.csv', 'data/y_train.csv', 'models/catboost_grid.pkl')
    get_catboost_random('data/X_train.csv', 'data/y_train.csv', 'models/catboost_random.pkl')
    get_lightgbm_bayesian('data/X_train.csv', 'data/y_train.csv', 'models/lightgbm_bayes.pkl')
    get_lightgbm_grid('data/X_train.csv', 'data/y_train.csv', 'models/lightgbm_grid.pkl')
    get_lightgbm_random('data/X_train.csv', 'data/y_train.csv', 'models/lightgbm_random.pkl')
    get_xgboost_bayesian('data/X_train.csv', 'data/y_train.csv', 'models/xgboost_bayes.pkl')
    get_xgboost_grid('data/X_train.csv', 'data/y_train.csv', 'models/xgboost_grid.pkl')
    get_xgboost_random('data/X_train.csv', 'data/y_train.csv', 'models/xgboost_random.pkl')
    
    # Evaluate
    get_evaluation(X_test_path='data/X_test.csv', y_test_path='data/y_test.csv', output_path='data/test_results.csv')
    
    #Store completion time
    end_workflow = time()

    # Prepare the output text
    output_text = (
        f"Start preprocessing: {datetime.fromtimestamp(start_preprocessing).strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"Start EDA: {datetime.fromtimestamp(start_eda).strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"Start modeling: {datetime.fromtimestamp(start_modeling).strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"End workflow: {datetime.fromtimestamp(end_workflow).strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        f"Preprocessing took: {start_eda - start_preprocessing:.2f} seconds\n"
        f"EDA took: {start_modeling - start_eda:.2f} seconds\n"
        f"Modeling took: {end_workflow - start_modeling:.2f} seconds\n"
        f"Total workflow time: {end_workflow - start_preprocessing:.2f} seconds\n"
    )

    # Save the output to a .txt file
    with open("workflow_times.txt", "w") as f:
        f.write(output_text)

    print("Workflow times have been saved to 'workflow_times.txt'")

if __name__ == "__main__":
    main()