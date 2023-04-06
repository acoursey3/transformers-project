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
We only consider accuracy as a metric in this project. Accuracy is defined in the typical way for binary classification in this model (see equation below.) We only chose accuracy due to the limited scope of this project, but more metrics (like ROC AUC, F1 score, etc.) should be considered in the future.

$$Accuracy = \frac{TP + TN}{TP + TN + FP + FN}$$

### Training Data
- Trained on [Fake or Real](https://ieeexplore.ieee.org/document/8906599) dataset
- Used `for-norm` version publicly available at [this link](https://bil.eecs.yorku.ca/datasets/) 
  - Balanced in terms of gender and class, preprocessed
- 53,866 audio samples of human utterances (real) or text-to-speech (fake)
- The dataset was designed with factors such as accent and gender in mind

### Evaluation Data
- Used same dataset as for training
- 10,798 samples in the validation split
- 4,643 samples in the test split (the test split is designed to be more difficult)

### Quantitative Analyses
See results section of README.md.

### Ethical Considerations
Since this model deals with human data, we do not recommend to use it outside of an academic setting. We do not consider our model to be ethical enough to use in the real world. Inaccurate or baised predictions could lead to unfair treatment of groups if this model was applied in a real world setting. We chose a dataset that considered these factors in an attempt to mitigate ethical issues, but we do not analyze the impact of our model on gender or accent.

### Caveats and Recommendations
We again recommend this model be used for academic purposes only. Further work can be done on this model to extend it to new domains and analyze the ethical impacts.
