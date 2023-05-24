# Link Adaptation codes
In this repository, we are going to explore the possibilities to improve the performance of link adaptation. Two aspects will be investigated, SINR prediction and CSI feedback compression.

## SINR prediction
USE LSTM and encode decoder to predictio the channel quality in terms of SINR

## CSI feedback
USE autoencoder to reduce the overhead of SINR transmission back to BS.
### Option 1
USE AE or VAE to transmit input of lstm and then use recovered signal as the input of LSTM prediction
### Option 2
USE AE or VAE to trainsmit the output of LSTM back to BS

## E2E model
Both options can be trained in a E2E manner.

## Other possible benchmarks
PCA for compression.

