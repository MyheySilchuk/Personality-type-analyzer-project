import data_processing as dp
import visualization as vis
import machine_learning as ml

def main():
    data_frame = dp.load_data(dp.file_path) 
    if data_frame is None:
        raise ValueError("Failed to load the dataset. Please check the file path and format.")

    cleaned_df = dp.cleaning_personality_dataset(data_frame)
    
    cleaned_df.to_csv('personality_dataset_cleaned.csv', index=False)
    
    print(cleaned_df.head())  # Display the first few rows of the cleaned DataFrame
    print(cleaned_df.info())  # Display DataFrame information to check data types and non-null counts
    
    print(cleaned_df.isnull().sum())  # Check for any remaining null values in the DataFrame"""

    #Visualisation of data we have
    vis.categorical_aspects(cleaned_df) #Fear stage & Social drain 
    vis.social_behavior(cleaned_df) #Social event attendance, Going outside & Post frequency
    vis.social_status(cleaned_df) #Friend circle' size & Time spending alone 
    
    #Learning model for personality analisation 
    data_for_model = ml.featuring_data_from_df(cleaned_df) # [X, y] 
    learning_model = ml.creating_model(data_for_model[0], data_for_model[1])
    ml.save_model(learning_model)
    ml.view_model_coefficients(learning_model, data_for_model[0])


if __name__ == "__main__":
    main()  # Run the main function to execute the data loading and cleaning process    

