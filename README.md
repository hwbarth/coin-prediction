# coin-prediction

https://data.binance.vision/ (Binance flat file for dev)

#**Hybrid two-stage model approach:**
use a deep neural network to estimate the residuals of the ARIMA model, with ARIMA capturing the linear patterns of the cryptocurrency price data while a deep neural network LSTM is used to model the remaining nonlinear patterns, thus improving the accuracy of forecasts. In addition, to mitigate the computational workload of the models, the LSTM model will run on a less frequent basis than the simpler RNN; this will also help eliminate frequency-generated noise in the LSTM's training data

Research:
https://www.sciencedirect.com/science/article/pii/S1057521923005719#:~:text=.%2C%202022).-,Khedr%20et%20al.,Ren%20et%20al.
https://www.sciencedirect.com/science/article/pii/S2405918818300928

Implied volatility (IV): to factor into Portfolio() equation
