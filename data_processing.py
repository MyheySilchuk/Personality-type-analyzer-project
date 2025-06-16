import pandas as pd

class personality_type:

    def __init__(self, name, value):
        self.name = name
        self.value = value

introvert = personality_type("introvert", 0)
extrovert = personality_type("extrovert", 1)


with open("file_path.txt", "r") as file:
    file_path = file.readline().strip()


def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None


def cleaning_personality_dataset(data_frame: pd.DataFrame) -> pd.DataFrame:
    data_frame = data_frame.copy()

    data_frame.columns = data_frame.columns.str.strip().str.lower().str.replace(' ', '_')
    data_frame.dropna(subset=['personality'], inplace=True)
    data_frame['personality'] = data_frame['personality'].map({'Introvert': introvert, 'Extrovert': extrovert})

    categorical_columns = data_frame.select_dtypes(include=['object']).columns.tolist()
    for column in categorical_columns:
        if column in data_frame.columns:
            data_frame[column] = data_frame[column].fillna(data_frame[column].mode()[0])
            if set(data_frame[column].dropna().unique()) <= {'Yes', 'No'}:
                data_frame[column] = data_frame[column].map({'Yes': 1.0, 'No': 0.0})

    numeric_columns = data_frame.select_dtypes(include=['float64', 'int64']).columns.tolist()
    for column in numeric_columns:
        if column in data_frame.columns:
            data_frame[column] = data_frame[column].fillna(data_frame[column].median())

    return data_frame
