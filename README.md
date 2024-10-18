# Diabetes Prediction App: https://diabetes-prediction999.streamlit.app/

This is a simple **Diabetes Prediction App** built using **Streamlit** and **Logistic Regression**. The app allows users to input their health data through a sidebar interface, and based on the data, it predicts whether the user is likely to have diabetes or not.

## Features

- User can input health parameters like:
  - Pregnancies
  - Glucose Level
  - Blood Pressure
  - Skin Thickness
  - Insulin Level
  - BMI (Body Mass Index)
  - Diabetes Pedigree Function
  - Age
- Model predicts if the person is diabetic based on the input values.
- Displays the accuracy of the model based on test data.

## Dataset

The app uses the **Pima Indians Diabetes Dataset**, which is a commonly used dataset for diabetes prediction. It consists of medical details and a target variable `Outcome`, indicating whether the patient is diabetic (1) or not (0).

## Installation

1. Clone the repository:
    ```bash
    git clone <repository-link>
    ```

2. Navigate into the project directory:
    ```bash
    cd diabetes-prediction-app
    ```

3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. Run the app:
    ```bash
    streamlit run app.py
    ```

## Model

- The app uses a **Logistic Regression** model, which is trained using the features from the dataset and tested with 20% of the data. 
- Model accuracy is displayed once the model is trained.

## Dependencies

The app requires Python 3.8 or higher and the following libraries:
- Streamlit
- Pandas
- NumPy
- Scikit-learn

Make sure to install all dependencies using the `requirements.txt` file.

## License

This project is open-source and free to use.