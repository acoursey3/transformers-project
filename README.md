# Should I use an RNN or a Transformer? An exploration of the impact of dataset size on audio classification.

Project for DS 5899 Spring 2023. By Cameron Baird and Austin Coursey.

## Overview
- Transformers becoming the state-of-the-art model for sequential data tasks
- In many domains, Recurrent Neural Networks like Long Short-Term Memory Networks (LSTMs) are still used
- Transformers are faster and do not suffer from vanishing gradient

*Are there any situations where an LSTM should be used?*

- Since Transformers are showing success in tasks with large language datasets:

**Research question:** *Will an LSTM outperform a Transformer model on audio classification when the dataset size is small?*

- We explore this question and find ...

## Method and Results

### Fake or Real Dataset
- Dataset of human (real) or text-to-speech (fake) speech utterances
- A few seconds
- Balanced gender and classes
- 53,866 samples in train, 10,798 in validation, 4,643 in test
- Test is designed to be harder and more realistic

**INSERT EXAMPLE WAV DATA PLOT HERE**

## Code Demo


## Model Card
The model card is stored in the [modelcard.md](modelcard.md) file. Please carefully read this model card before attempting to use our pretrained models.

## Critical Analysis


## Resource Links

