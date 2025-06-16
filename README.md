AI-powered Personality Profiler
Author: Mihaliss
Version: 1.0

Date: 6/16/2025

🧠 Description
This project offers an insightful analysis of behavioral data to predict personality type: Introvert or Extrovert. 
It leverages a logistic regression model for classification and features a clean, interactive user interface built with Streamlit, making the insights easily accessible.

📁 Project Structure
The project is organized into modular components to ensure clarity, maintainability, and ease of understanding:

- data_processing.py: Handles the loading, cleaning, and preparation of raw behavioral data for machine learning.

- visualization.py: Contains functions to create compelling visualizations, helping to explore patterns and relationships within the personality-related behavioral data.

- machine_learning.py: Focuses on training the logistic regression model, evaluating its performance, and managing model persistence (saving and loading) using joblib.

- app.py: The core Streamlit application, providing a web-based interface for users to upload data, view predictions, and interact with the model.

- main.py: The main entry point for running different modules or the entire machine learning pipeline.

- file_path.txt: Stores data path for the project

🚀 Getting Started
Follow these simple steps to get the AI-powered Personality Profiler up and running on your local machine.

# Install data
Place your personality_dataset.csv file in the appropriate directory and copy its path into file_path.txt.

# Install dependencies
pip install -r requirements.txt

# Run main.py 
If you want to see data info and graphics 

# Run app
Once the dependencies are installed, you can launch the interactive web application:

streamlit run app.py


🧰 Requirements
This project requires Python 3.8+ and the following libraries, which are listed in requirements.txt:

pandas

numpy

matplotlib (or seaborn, depending on implementation)

scikit-learn

streamlit

joblib



For easy testing, consider adding a small sample behavioral_data.csv to a data/ directory in your repository.

📄 License
This project is intended for educational purposes and is open to modification and reuse.

✨ Optional
Want to contribute or adapt this project to your specific needs? Feel free to fork the repository, experiment with the code, and make it your own!
