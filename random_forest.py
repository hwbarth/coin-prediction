# random forest model for predicting period return class (1-10)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import argparse
from datetime import datetime
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature


def getData(df, target):

    data = df.drop(columns = ['open', 'high', 'low', 'close', 'volume'])
    data.dropna(how='any', inplace=True)
    features = [c for c in data.columns if "return" not in c]
    X = data[features]
    y = data[['timestamp', target]]
    X.head()

    X.reset_index(inplace=True)
    X.drop(columns=['index'], inplace=True)

    y.reset_index(inplace=True)
    y.drop(columns=['index'], inplace=True)


    timestamps = X['timestamp'].values  # Extract timestamp
    features = X.drop(columns=['timestamp'])

    # Step 2: Apply StandardScaler
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    # Step 3: Combine the scaled features with the timestamp
    scaled_df = pd.DataFrame(scaled_features)

    scaled_df = pd.concat([y[target], scaled_df], axis=1)
    scaled_df = pd.concat([X['timestamp'], scaled_df], axis=1)

    # Use qcut to divide 'returns' into 11 equal-sized groups and label them from 1 to 11
    scaled_df['return_group'] = pd.qcut(scaled_df[target], q=11, labels=range(1, 11 + 1))



    train_size = 0.8  # 80% for training and 20% for testing

    split_index = int(len(scaled_df) * train_size)

    train_df = scaled_df[:split_index]
    test_df = scaled_df[split_index:]

    X_train = train_df.drop(columns=[target, 'return_group']).to_numpy()
    y_train = train_df[['timestamp','return_group']].to_numpy()

    X_test = test_df.drop(columns=[target, 'return_group']).to_numpy()
    y_test = test_df[['timestamp','return_group']].to_numpy()

    TRAIN_TIMESTAMPS = X_train[:, 0]

    TEST_TIMESTAMPS = X_test[:, 0]

    #remove timestamps
    X_train = X_train[:, 1:]
    y_train = y_train[:, 1:].flatten()

    X_test = X_test[:, 1:]
    y_test = y_test[:, 1:].flatten()

    return X_train, y_train, X_test, y_test



def getReports(y_test, y_pred, model_name, resid=None, output_folder='visualizations'):

    output_folder = f"{output_folder}/random-forest/{model_name}" 
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Calculate accuracy for Random Forest
    accuracy_rf = accuracy_score(y_test, y_pred)
    print(f'Random Forest Accuracy: {accuracy_rf:.2f}')

    # Generate confusion matrix for Random Forest
    conf_matrix_rf = confusion_matrix(y_test, y_pred)

    # Classification report for Random Forest
    class_report_rf = classification_report(y_test, y_pred)
    print("Random Forest Classification Report:\n", class_report_rf)

    # Plotting confusion matrix for Random Forest
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix_rf, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
    plt.title("Random Forest Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(os.path.join(output_folder, 'confusion_matrix_rf.png'), bbox_inches='tight')
    plt.close()

    # Generate classification report as a DataFrame for visualization
    class_report = classification_report(y_test, y_pred, output_dict=True)
    class_labels = [key for key in class_report.keys() if key not in ('accuracy', 'macro avg', 'weighted avg')]
    precision = [class_report[label]['precision'] for label in class_labels]
    recall = [class_report[label]['recall'] for label in class_labels]
    f1 = [class_report[label]['f1-score'] for label in class_labels]

    # Create a DataFrame for the classification report
    class_report_df = pd.DataFrame({
        'Class': class_labels,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1
    })
    class_report_df.set_index('Class', inplace=True)

    # Plotting the classification report
    plt.figure(figsize=(10, 6))
    class_report_df.plot(kind='bar', alpha=0.75, ax=plt.gca())
    plt.title("Classification Report")
    plt.ylabel("Score")
    plt.xticks(rotation=0)
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(output_folder, 'classification_report.png'), bbox_inches='tight')
    plt.close()

    # Plotting the distribution of 'returns'
    plt.figure(figsize=(10, 6))
    sns.histplot(y_pred, kde=True, bins=30, color='skyblue')
    plt.title("Distribution of 'Returns'")
    plt.xlabel("Returns")
    plt.ylabel("Frequency")
    plt.savefig(os.path.join(output_folder, 'distribution_returns.png'), bbox_inches='tight')
    plt.close()

    # Calculate correct group prediction accuracy
    correct_predictions = (y_pred == y_test).sum()
    total_predictions = len(y_test)
    accuracy_group = correct_predictions / total_predictions
    print(f"Correct Group Prediction Accuracy: {accuracy_group:.2f}")

    # Plotting the distribution of 'resid' if provided
    if resid is not None:
        plt.figure(figsize=(10, 6))
        sns.histplot(resid, kde=True, bins=30, color='skyblue')
        plt.title("Distribution of Residuals")
        plt.xlabel("Residual Values")
        plt.ylabel("Frequency")
        plt.axvline(x=resid.mean(), color='r', linestyle='--', label='Mean')
        plt.axvline(x=resid.median(), color='g', linestyle='--', label='Median')
        plt.legend()
        plt.savefig(os.path.join(output_folder, 'distribution_residuals.png'), bbox_inches='tight')
        plt.close()



class rfModel:
    def __init__(self, X_train, y_train, X_test, y_test, ticker, frequency, target, timestamps, model_name):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.ticker = ticker
        self.frequency = frequency
        self.target = target
        self.timestamps = timestamps
        self.model_name = model_name

    def fitModelAndPredict(self):
        # Set up MLflow tracking
        mlflow.start_run()

        # Initialize and fit the model
        self.model = RandomForestClassifier(n_estimators=1000, random_state=42)
        self.model.fit(self.X_train, self.y_train)

        # Predict and evaluate for Random Forest
        self.y_pred = self.model.predict(self.X_test)
        accuracy_rf = accuracy_score(self.y_test, self.y_pred)

        # Log model and metrics with MLflow
        model_date = datetime.now().date().strftime("%Y-%m-%d")
        
        mlflow.log_param("ticker", self.ticker)
        mlflow.log_param("frequency", self.frequency)
        mlflow.log_param("target", self.target)
        mlflow.log_param("model_date", model_date)
        mlflow.log_metric("accuracy", accuracy_rf)
        
        # Log the model
        mlflow.sklearn.log_model(self.model, "model")

        # Print results
        print(f'Random Forest Accuracy: {accuracy_rf}')
        print(classification_report(self.y_test, self.y_pred))

        # End the MLflow run
        mlflow.end_run()

        # cutoff = max((len(self.timestamps), len(self.y_pred), len(self.y_test)))

        # t = self.timestamps[:cutoff]
        # p = self.y_pred[:cutoff]
        # te = self.y_test[:cutoff]

        # data = {
        #     'timestamp': t,
        #     'predicted': p,
        #     'real': te
        # }

        # self.results = pd.DataFrame(data)

        # self.results.to_csv(f"models/model-output/random-forest/{self.model_name}.csv")

    def setModel(self, model):
        self.model = model



if __name__ == "__main__":

    #default
    ticker = "XBTUSD"
    frq = "1"
    target = "return_16n"
    parser = argparse.ArgumentParser(description='Process cryptocurrency data.')
    parser.add_argument('ticker', type=str, help='The cryptocurrency ticker symbol (e.g., BTC, ETH)')
    parser.add_argument('frequency', type=int, help='The frequency of data points (e.g., 1 for daily, 7 for weekly)')
    parser.add_argument('target', type=str, help='target return period (e.g return_n8 (return for period n + 8), 2^i')


    args = parser.parse_args()
    ticker = args.ticker
    frq = args.frequency
    target = args.target

    df = pd.read_csv(f"data/silver_prices/{ticker}_{frq}_silver.csv")
    print(df)
    timestamps = list(df['timestamp'])
    X_train, y_train, X_test, y_test = getData(df, target)


    model_date = datetime.now().date().strftime("%Y-%m-%d")

    # model_path = f'models/{ticker}_{frq}_model_{model_date}.h5'
    model_name = f"{ticker}_{frq}_{target}_{model_date}"
    model_path = "models/random-forest/"+model_name+'.joblib'

    rf = rfModel(X_train, y_train, X_test, y_test, ticker, frq, target, timestamps, model_name)

    rf.fitModelAndPredict()

    getReports(rf.y_test, rf.y_pred, rf.model_name)

 

    # After your model training and before logging the model
    signature = infer_signature(rf.X_train, rf.model.predict(rf.X_train))

    # with mlflow.start_run():
    #     mlflow.keras.log_model(rf.model, "model", signature=signature)
    #     # Save the model locally
    #     rf.model.save(f"models/random-forest/{ticker}_{frq}_{target}_{model_date}.h5")



import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
import joblib


# Start MLflow run
with mlflow.start_run():
    # Log model parameters, metrics, etc.
    mlflow.log_param("n_estimators", rf.n_estimators)
    mlflow.log_param("max_depth", rf.max_depth)

    # Log the scikit-learn model
    mlflow.sklearn.log_model(rf, "model")

    # Optionally save the model locally using joblib
    # model_path = f"models/{model_name}.joblib"
    joblib.dump(rf, model_path)
    mlflow.log_artifact(model_path)  # Log the model artifact

    # with mlflow.start_run():
    #     mlflow.keras.log_model(rf.model, "model", signature=signature)
    #     # Save the model locally
    #     rf.model.save(f"models/random-forest/{ticker}_{frq}_{target}_{model_date}.h5")








