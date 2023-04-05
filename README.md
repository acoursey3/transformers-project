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

### Training
- Consider 4 training dataset splits
  - 25%, 50%, 75%, and 100% data available 
- Encoder Transformer with binary classification head
  - Input: entire raw audio sequence
  - Use 1D Convolutional layers to reduce the sequence length to 512
  - Learned positional encodings
  - Trained until validation accuracy stopped improving
  - Fixed hyperparameters
  - More details in code demo
  - **INSERT PARAM COUNT** parameters here
  - **INSERT INFERENCE TIME**
- LSTM
  - Input: first 2 seconds of data (memory issues)
  - Binary classification on output of LSTM
  - Trained for 10 epochs (time issues)
  - Fixed hyperparameters
  - **INSERT PARAM COUNT**  
  - **INSERT INFERENCE TIME**

### Results
**INSERT RESULS HERE**

## Code Demo
**ADD MODEL ARCHITECTURE IN DEMO**

## Model Card
The model card is stored in the [modelcard.md](modelcard.md) file. Please carefully read this model card before attempting to use our pretrained models.

## Critical Analysis
- In this project, we learned ... **INSERT IMPACT OF PROJECT AND WHAT IT REVEALS**
- We wrote code for Tree-Structured Parzen Estimator hyperparameter optimization, but computation time was an issue. The hyperparameters may not be optimal, so our results might change with optimal hyperparameters.
- We did not perform any gender/accent bias analysis
- Limited evaluation metrics
- This dataset may not be representative of their performance on other tasks

**Next Steps**
- Optimize hyperparameters
- Explore more datasets (different domains and tasks)
- Calculate more evaluation metrics

## Resource Links
- FOR paper
- FOR download link
- LSTM explanation
- Paper comparing transformer and lstm
- TPE hyperparameter optimization
- Hackers clone CEO voice
