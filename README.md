🌱 **AI-Based Crop Recommendation System Using Machine Learning**

📌 **Project Overview:**

The AI-Based Crop Recommendation System is a machine learning project designed to help farmers and agricultural professionals identify suitable crops based on soil nutrient levels and environmental conditions. The system analyzes essential agricultural parameters, including Nitrogen (N), Phosphorus (P), Potassium (K), temperature, humidity, soil pH, and rainfall, to predict the most appropriate crop for the given conditions.

The project uses Python and machine learning classification algorithms to analyze agricultural data and generate crop recommendations. A Flask-based web application provides a simple, user-friendly interface through which users can enter soil and environmental parameters and receive a predicted crop.

The main objective of this project is to demonstrate how Artificial Intelligence (AI) and Machine Learning (ML) can support data-driven agricultural decisions and contribute to more informed crop selection.

🎯 **Project Objectives:**

1.Develop a machine learning model that predicts suitable crops from agricultural input parameters.

2.Analyze soil nutrients and environmental conditions to identify patterns associated with different crops.

3.Compare multiple classification models and select the model with the highest test accuracy.

4.Build an interactive web application using Flask for crop prediction.

5.Perform Exploratory Data Analysis (EDA) to understand the dataset and visualize important relationships.

6.Demonstrate the practical application of machine learning in agriculture.

🛠️ **Technologies Used:**

i.**Python**: Core programming language used for data processing, model training, and prediction.

ii.**Pandas**: Used to load, clean, and manipulate the agricultural dataset.

iii.**NumPy**: Used for numerical operations and preparing input features for predictions.

iv.**Scikit-learn**: Used to train, evaluate, compare, and select machine learning classification models.

v.**Matplotlib**: Used to create charts and visualize dataset characteristics.

vi.**Seaborn**: Used to generate statistical visualizations, including heatmaps, boxplots, and distribution plots.

vii.**Flask**: Used to develop the web application and process user-submitted crop prediction inputs.

viii.**HTML**: Used to structure the web interface.

ix.**Pickle**: Used to save and load the trained machine learning model.

x.**Git and GitHub**: Used for source code management and project version control.

📊 **Dataset Description:**

The project uses Crop.csv, a dataset containing agricultural measurements and crop labels. The model uses seven numerical input features to predict the crop category.

**Input Features:**
**Feature   **                                        **  Description**

Nitrogen (N)                                     Nitrogen content in the soil.

Phosphorus (P)                                   Phosphorus content in the soil.

Potassium (K)                                    Potassium content in the soil.

Temperature                                      Temperature associated with the agricultural conditions.

Humidity                                         Humidity level associated with the agricultural conditions.

Soil pH                                          Measurement indicating the acidity or alkalinity of the soil.

Rainfall                                         Rainfall measurement associated with the agricultural conditions.

**Target variable**: label — the crop category that the machine learning model learns to predict.

The dataset is loaded using Pandas, and rows containing missing values are removed before the model-training process. The cleaned data is then divided into input features and target labels.

⚙️ **Project Workflow:**

1. **Data Loading and Preprocessing:**

The dataset is loaded from Crop.csv using Pandas. The program examines the dataset's dimensions, column names, crop categories, and missing values. Records containing missing values are removed to prepare the data for analysis and model training.

2. **Exploratory Data Analysis (EDA):**

Exploratory Data Analysis is performed to understand the dataset and identify patterns in agricultural parameters. The project generates several visualizations, including:

i.Crop distribution charts to examine the frequency of crop categories.

ii.Correlation heatmaps to study relationships between numerical features.

iii.Histograms to visualize feature distributions.

iv.Boxplots to compare nitrogen, phosphorus, and potassium values across selected crops.

v.Violin plots to explore rainfall distributions across selected crop categories.

vi.Scatterplots to visualize relationships between temperature, humidity, soil pH, and rainfall.

vii.These visualizations help explore the characteristics of the dataset before training the models.

3. **Data Splitting:**

The dataset is divided into training and testing sets using an 80:20 split. The training data is used to fit the machine learning models, while the testing data is used to evaluate their predictive performance on held-out records.

Stratified splitting is applied to preserve the relative distribution of crop categories across the two sets.

4. **Machine Learning Model Development:**

The project evaluates multiple classification approaches to identify a suitable crop prediction model.

**Random Forest Classifier**: The baseline model uses an ensemble of decision trees to predict crop categories from the seven input features.

**Tuned Random Forest**: RandomizedSearchCV is used to explore selected hyperparameter combinations when the baseline model's accuracy is below the configured threshold.

**Voting Classifier**: A hybrid ensemble combines the predictions of Random Forest, Gradient Boosting, and K-Nearest Neighbors using hard voting.

**Stacking Classifier**: A hybrid ensemble combines Random Forest, Gradient Boosting, and K-Nearest Neighbors, with Logistic Regression serving as the final estimator.

The models are trained and evaluated to compare their predictive performance.

5. **Model Evaluation:**

The project evaluates classification performance using the following metrics and techniques:

**Accuracy Score:** Measures the proportion of test samples predicted correctly.

**Classification Report:** Provides precision, recall, and F1-score for each crop category.

**Cross-Validation:** Evaluates the baseline Random Forest model across five folds to examine performance across different data splits.

**Confusion Matrix:** Shows correct and incorrect predictions for the crop categories.

**Feature Importance:** Displays the relative importance assigned to input features by the selected Random Forest model.

The model with the highest recorded test accuracy among the evaluated candidates is selected for saving. The reported accuracy should be obtained by running the training script; no specific accuracy value is assumed in this README.

6. **Model Saving and Integration:**

The selected model is serialized into best_crop_model.pkl using Pickle. The Flask application loads this saved model when the application starts.

This separation allows the web application to use the trained model for predictions without retraining the model every time a user submits a request.

7. **Web Application and Crop Prediction:**

The Flask application provides a web interface for entering the required agricultural parameters. When the user submits the form, the application:

1.Collects the seven input values.

2.Checks that the required fields contain numeric values.

3.Performs basic validation on the pH, humidity, and rainfall inputs.

4.Organizes the inputs into the feature order expected by the trained model.

5.Uses the saved model to predict a crop category.

6.Displays the predicted crop on the web interface.

The application also displays validation errors when the submitted inputs are missing or invalid.
