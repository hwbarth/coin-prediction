#lstm module
from sklearn.preprocessing import MinMaxScaler
import pandas as pd 
import numpy as np
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Activation, Input
from mlflow.models import infer_signature
import mlflow
import mlflow.keras
from datetime import datetime
import argparse
from keras.optimizers import Adam
import os 
from sklearn.metrics import accuracy_score, classification_report
# from sklearn.metrics import accuracy_score, classification_report


class Lstm:
    def __init__(self, data, ticker, frequency, target, timestamps, model_name, SEQ_LENGTH=100):
        self.data = data
        self.ticker = ticker
        self.frequency = frequency
        self.target = target
        self.timestamps = timestamps
        self.model_name = model_name
        self.SEQ_LENGTH = SEQ_LENGTH
    
    def getData(self):

        df = self.data
        scaler = MinMaxScaler()
        self.scaler = scaler
        close_price = df[self.target].values.reshape(-1, 1)

        scaled_close = scaler.fit_transform(close_price)
        # SEQ_LEN = 100
        self.X_train, self.y_train, self.X_test, self.y_test = preprocess(scaled_close, self.SEQ_LENGTH, train_split = 0.95)


    # def compileModel(self, opt='adam', loss='mean_squared_error'):
    #     #We’re creating a 3 layer LSTM Recurrent Neural Network. We use Dropout with a rate of 20% to combat overfitting during training:

    #     # Set your constants
    #     DROPOUT = 0.2 
    #     WINDOW_SIZE = self.SEQ_LENGTH - 1

    #     # Define the model
    #     self.model = Sequential()

    #     self.model.add(Bidirectional(
    #         LSTM(WINDOW_SIZE, return_sequences=True),  # Use LSTM directly
    #         input_shape=(WINDOW_SIZE, self.X_train.shape[-1])
    #     ))
    #     self.model.add(Dropout(rate=DROPOUT))

    #     # Add more layers as needed...
    #     self.model.add(Dense(1))  # Example output layer

    #     # Compile the model
    #     self.model.compile(optimizer=opt, loss=loss)

    #     # Summary of the model
    #     self.model.summary()

    #     self.model.add(Activation('linear'))

    def compileModel(self, opt='adam', loss='mean_squared_error', learning_rate=0.0001):
        # Set your constants
        DROPOUT = 0.2 
        WINDOW_SIZE = self.SEQ_LENGTH - 1

        # Define the model
        self.model = Sequential()

        # Add Input layer
        self.model.add(Input(shape=(WINDOW_SIZE, self.X_train.shape[-1])))

        # Add LSTM layer with Bidirectional wrapper
        self.model.add(Bidirectional(LSTM(WINDOW_SIZE, return_sequences=True)))

        # Add Dropout layer
        self.model.add(Dropout(rate=DROPOUT))

        # Add more layers as needed...
        self.model.add(Dense(1))  # Example output layer

        # Define the optimizer with the specified learning rate
        if opt == 'adam':
            optimizer = Adam(learning_rate=learning_rate)
        else:
            optimizer = opt  # Use other optimizers if specified

        # Compile the model
        self.model.compile(optimizer=optimizer, loss=loss)

        # Add Activation layer (if necessary, usually added in the Dense layer for the output)
        self.model.add(Activation('linear'))  # This line might not be needed since Activation can be in Dense

        # Summary of the model
        print(self.model.summary())

    # def trainModel(self):

    #     self.today = datetime.now().strftime("%Y-%m-%d")

    #     # Set the experiment name (optional)
    #     mlflow.set_experiment(f"{self.ticker}_{self.frequency}_{self.target}_experiment")

    #     BATCH_SIZE = 32

    #     with mlflow.start_run():
    #         # Compile the model
    #         # self.model.compile(
    #         #     loss='mean_squared_error',
    #         #     optimizer='adam'
    #         # )

    #         # Log parameters
    #         mlflow.log_param("batch_size", BATCH_SIZE)
    #         mlflow.log_param("epochs", 15)
    #         mlflow.log_param("optimizer", "adam")

    #         history = self.model.fit(
    #             self.X_train,
    #             self.y_train,
    #             epochs=15,
    #             batch_size=BATCH_SIZE,
    #             shuffle=False,
    #             validation_split=0.1
    #         )

    #         # Log metrics
    #         mlflow.log_metric("final_loss", history.history['loss'][-1])
    #         mlflow.log_metric("final_val_loss", history.history['val_loss'][-1])

    # def saveModel(self):

    #     # After your model training and before logging the model
    #     signature = infer_signature(self.X_train, self.model.predict(self.X_train))

    #     with mlflow.start_run():
    #         mlflow.keras.log_model(self.model, "model", signature=signature)
    #         # Save the model locally
    #         self.model.save(f"models/lstm/{self.ticker}_{self.frequency}_{self.target}_{self.today}.h5")
    def trainModel(self):
        self.model_date = datetime.now().date().strftime("%Y-%m-%d")

        # Set the experiment name (optional)
        mlflow.set_experiment(f"{self.ticker}_{self.frequency}_{self.target}_{self.model_date}_experiment")

        BATCH_SIZE = 32

        with mlflow.start_run():
            # Log parameters
            mlflow.log_param("batch_size", BATCH_SIZE)
            mlflow.log_param("epochs", 15)
            mlflow.log_param("optimizer", "adam")

            # Train the model
            history = self.model.fit(
                self.X_train,
                self.y_train,
                epochs=15,
                batch_size=BATCH_SIZE,
                shuffle=False,
                validation_split=0.1
            )

            # Log metrics
            mlflow.log_metric("final_loss", history.history['loss'][-1])
            mlflow.log_metric("final_val_loss", history.history['val_loss'][-1])

    def saveModel(self):
        # Predict on validation/test data
        y_pred = self.model.predict(self.X_test)
        # Convert predictions to class labels if necessary
        y_pred_classes = (y_pred > 0.5).astype(int)  # Example for binary classification

        # Calculate accuracy (or any other relevant metrics)
        accuracy = accuracy_score(self.y_test, y_pred_classes)

        # Create a model directory
        model_name = f"{self.ticker}_{self.frequency}_{self.target}_{self.model_date}"
        model_dir = f"models/lstm/{model_name}"
        os.makedirs(model_dir, exist_ok=True)

        # Save the model as an .h5 file
        model_path = os.path.join(model_dir, f"{model_name}.h5")
        self.model.save(model_path)

        # Log the model and metrics with MLflow
        with mlflow.start_run():
            # mlflow.keras.log_model(self.model, "model")
            mlflow.log_metric("accuracy", accuracy)

            # Save metrics to a text file
            metrics_path = os.path.join(model_dir, "metrics.txt")
            with open(metrics_path, "w") as f:
                f.write(f"Final Accuracy: {accuracy}\n")
                f.write("Classification Report:\n")
                f.write(classification_report(self.y_test, y_pred_classes))

        print(f'LSTM Model saved to {model_path} with accuracy: {accuracy}')

    def predictAndUnscale(self):
        self.y_pred = self.model.predict(self.X_test)
        # Select the last timestep
        # Assuming y_hat has shape (batch_size, timesteps, features)
        # Select the last timestep for predictions
        self.y_pred_last_timestamp = self.y_pred[:, -1, :]  # Make sure y_hat has the expected shape

        # Inverse transform the predictions
        self.y_pred_inverse = self.scaler.inverse_transform(self.y_pred_last_timestamp)

        # Assuming y_test is a 2D array (if you're using a sequence of values)
        # If y_test was reshaped correctly when prepared, do this:
        self.y_test_inverse = self.scaler.inverse_transform(self.y_test)

        if self.y_test.ndim == 3:
            self.y_test_last_timestep = self.y_test[:, -1, :]  # Only if y_test is 3D
            self.y_test_inverse = self.scaler.inverse_transform(self.y_test_last_timestep)

    def setModel(self, model):
        self.model = model

        

def to_sequences(data, seq_len):
    d = []

    for index in range(len(data) - seq_len):
        d.append(data[index: index + seq_len])

    return np.array(d)

def preprocess(data_raw, seq_len, train_split):

    data = to_sequences(data_raw, seq_len)

    num_train = int(train_split * data.shape[0])

    X_train = data[:num_train, :-1, :]
    y_train = data[:num_train, -1, :]

    X_test = data[num_train:, :-1, :]
    y_test = data[num_train:, -1, :]

    return X_train, y_train, X_test, y_test

if __name__ == "__main__":
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

    df = df[-int((len(df)/2)):]

    print(df)
    timestamps = list(df['timestamp'])
    model_date = datetime.now().date().strftime("%Y-%m-%d")
    #
    model_name = f"{ticker}_{frq}_{target}_{model_date}"
    model_path = "models/random-forest"+model_name

    lstm = Lstm(df, ticker, frq, target, timestamps, model_name, SEQ_LENGTH=100)

    lstm.getData()
    lstm.compileModel()
    lstm.trainModel()
    lstm.saveModel()
    lstm.predictAndUnscale()
    preds = lstm.y_pred_inverse
    real = lstm.y_test_inverse
    data = {
        "timestamps": timestamps,
        "predicted":preds,
        "real":real,
    }
    result = pd.DataFrame(data)
    result.to_csv(f"models/model-output/lstm/{model_name}.csv")
    # import tensorflow as tf

    # # List all physical devices
    # physical_devices = tf.config.list_physical_devices('GPU')
    # print("GPUs available: ", physical_devices)

    # # Optional: Set memory growth
    # if physical_devices:
    #     tf.config.experimental.set_memory_growth(physical_devices[0], True)
