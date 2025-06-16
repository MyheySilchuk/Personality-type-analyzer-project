import data_processing as dp
import pandas as pd
import numpy as np
import joblib
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

def featuring_data_from_df(data_frame: pd.DataFrame):

    features_for_model = data_frame[['time_spent_alone', 'stage_fear', 'social_event_attendance',
                        'going_outside', 'drained_after_socializing', 'friends_circle_size',
                        'post_frequency']]
        

    target_variable = data_frame['personality'].apply(lambda x: x.value)  # Convert personality_type to numeric values
        
    return [features_for_model, target_variable]

def creating_model(X, y):
    """
    Parameters:
    X (pd.DataFrame): Features for the model.
    y (pd.Series): Target variable.
    
    Returns:
    model: A trained logistic regression model.
    """
    # Create a pipeline with scaling + logistic regression
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=0.1,                # weaker regularization
            penalty='l2',         # L2 regularization (default)
            class_weight='balanced',  # compensate for class imbalance
            solver='lbfgs',       # recommended for L2
            max_iter=1000         # sometimes need to increase iterations
        )
    )

    # Evaluate the model using 5-fold cross-validation
    scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    model.fit(X, y)  # Train the model on all data

    print("Cross-validation scores:", scores)
    print("Average Accuracy:", scores.mean())

    return model


def save_model(model, filename='personality_type_analisation_model.pkl'):
    """
    Save the trained model to a file.
    
    Parameters:
    model: The trained model to save.
    filename (str): The name of the file to save the model to.
    """
    joblib.dump(model, filename)
    print(f"Model saved to {filename}")


def view_model_coefficients(model, X):
    log_reg = model.named_steps['logisticregression']
    # Get feature names
    feature_names = X.columns
    # Coefficients
    coefficients = log_reg.coef_[0]  # for binary classification – this is 1 array
    # Create a table with results
    coef_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coefficients,
        'Importance (abs)': np.abs(coefficients)
    }).sort_values(by='Importance (abs)', ascending=False)

    print(coef_df)

