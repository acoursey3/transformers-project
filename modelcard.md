# Model Card

### Model Details
- Developed by Cameron Baird and Austin Coursey for Vanderbilt University's DS 5899
- Model Date: 4/5/2023
- Encoder Transformer with binary classification head
- Long Short-Term Memory Network (LSTM)
- Model training code in this repository
- Hyperparameters were manually tuned
- With questions, please raise an issue in this repository
- No license

### Intended Use
These models are intended to be used by researchers or students interested in exploring the performance differences between LSTMs and Transformers in audio classification tasks. It is not intended to be used for real-world applications. Any real world applications would be considered out of scope.

### Factors
As the model is used for speech classification, factors such as gender, race, or accent may be relevant. We chose a dataset that attempted to be diverse in these categories, but there could still be biases in the dataset or models. We do not evaluate the performance of our model on these different factors. For that reason, we advise against a real-world application of our models.

### Metrics


### Evaluation Data
10798 validation
4643 test (text to speech higher quality for this)

### Training Data
53866 samples (about half)
Used for norm partition (sampling rate normalized, 1 audio channel, we don't expect it changes anything)

### Quantitative Analyses


### Ethical Considerations


### Caveats and Recommendations

