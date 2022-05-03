from keras.models import Sequential
from keras.layers import Dense, GRU, LSTM
import matplotlib.pyplot as plt
from numpy import array
import pandas as pd
from sklearn.model_selection import train_test_split

def split_sequence(sequence, n_steps):
	X, y = list(), list()
	for i in range(len(sequence)):
		end_ix = i + n_steps
		if end_ix > len(sequence)-1:
			break
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)

def build_rnn(k, n_features, recurrent_layer):
    model = Sequential()
    model.add(recurrent_layer(50, activation="relu", input_shape=(k, n_features)))
    model.add(Dense(1))
    return model

def main(k):
    title = "k = " + str(k)
    print(title)

    # Reshape data
    X, y = split_sequence(raw_seq, k)
    n_features = 1
    X = X.reshape((X.shape[0], X.shape[1], n_features))

    # Split data
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Build models
    lstm = build_rnn(k, n_features, LSTM)
    gru = build_rnn(k, n_features, GRU)

    # Train models
    lstm.compile(optimizer="adam", loss="mse", metrics=["mape"])
    history_lstm = lstm.fit(x_train, y_train, epochs=200, verbose=0)
    gru.compile(optimizer="adam", loss="mse", metrics=["mape"])
    history_gru = gru.fit(x_train, y_train, epochs=200, verbose=0)

    # Plot performances
    fig = plt.figure(figsize=(12.8, 4.8))
    fig.add_subplot(121)
    plt.plot(history_lstm.history["loss"], label="LSTM")
    plt.plot(history_gru.history["loss"], label="GRU")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.legend()
    fig.add_subplot(122)
    plt.plot(history_lstm.history["mape"], label="LSTM")
    plt.plot(history_gru.history["mape"], label="GRU")
    plt.xlabel("Epoch")
    plt.ylabel("MAPE")
    plt.legend()
    fig.suptitle(title)
    # plt.show()
    plt.savefig("images/" + title + ".png")
    plt.close(fig)

    # Print test accuracy
    lstm_mse, lstm_mape = lstm.evaluate(x_test, y_test, verbose=2)
    gru_mse, gru_mape = gru.evaluate(x_test, y_test, verbose=2)

if __name__ == "__main__":
    # Load data
    data = pd.read_csv("daily-min-temperatures.csv")
    raw_seq = data["Temp"].values

    # Train RNNs
    # main(3)
    # main(5)
    # main(7)
    # main(9)
