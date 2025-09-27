# Diabetes Detector

This project is a machine learning model built using TensorFlow and scikit-learn to predict whether a person has diabetes based on their health data.

## Dataset
The project uses a dataset named `diabetes_dataset_2000.csv` which includes the following features:
- Age
- BMI (Body Mass Index)
- Blood Pressure
- Glucose
- Family History (1 = Yes, 0 = No)

The target column is:
- Diabetes (1 = Positive, 0 = Negative)

## How it Works
1. The dataset is loaded and split into training and testing sets.
2. Features are standardized using `StandardScaler`.
3. A neural network is built with TensorFlow Keras:
   - Input layer with 5 features
   - Hidden layers with 16 and 8 neurons (ReLU activation)
   - Output layer with sigmoid activation for binary classification
4. The model is trained for 50 epochs with a batch size of 16.
5. Accuracy is evaluated on the test set.
6. A prediction is demonstrated on a sample new patient.

## Requirements
Install the following dependencies before running the project:
```bash
pip install pandas numpy tensorflow scikit-learn
```

## Usage
1. Place the `diabetes_dataset_2000.csv` file in the same directory as `diabetes_detector.py`.
2. Run the script:
```bash
python diabetes_detector.py
```
3. The script will train the model, evaluate accuracy, and provide a prediction for a sample patient.

## Example Output
```
Model Accuracy: 85.00%
Predicted diabetes chance: 72.45%
```

## File Structure
```
├── diabetes_detector.py
├── diabetes_dataset_2000.csv
├── README.md
```

## Future Improvements
- Use a larger and more diverse dataset.
- Implement hyperparameter tuning.
- Save and load the trained model using TensorFlow SavedModel format.
- Build a web or GUI interface for easier predictions.
