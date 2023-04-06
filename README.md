# Should I use an RNN or a Transformer? An exploration of the impact of dataset size on audio classification.

Project for DS 5899 Spring 2023. By Cameron Baird and Austin Coursey.

## Overview
- Transformers becoming the state-of-the-art model for sequential data tasks
- In many domains, Recurrent Neural Networks like Long Short-Term Memory Networks (LSTMs) are still used
- Transformers are faster and do not suffer from vanishing gradient

*Are there any situations where an LSTM should be used?*

- Since Transformers are showing success in tasks with large language datasets:

**Research question:** *Will an LSTM outperform a Transformer model on audio classification when the dataset size is small?*

- We explore this question by training LSTMs and Transformers on an audio classification dataset of varying sizes.
- We find that the Transformer model outperforms the LSTM in terms of generalization abilities regardless of the dataset size.

## Method and Results

### Fake or Real Dataset
- Dataset of human (real) or text-to-speech (fake) speech utterances
- A few seconds
- Balanced gender and classes
- 53,866 samples in train, 10,798 in validation, 4,643 in test
- Test is designed to be harder and more realistic

![Example input waveforms for our networks.](https://github.com/acoursey3/transformers-project/blob/main/pics/audio_sample.png?raw=true)

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
  - 268,732,673 total parameters
- LSTM
  - Input: first 2 seconds of data (memory issues)
  - Binary classification on output of LSTM
  - Trained for 10 epochs (time issues)
  - Fixed hyperparameters
  - 2,065,153 total parameters 

### Results

1. Training

![Results from our training procedure.](https://github.com/acoursey3/transformers-project/blob/main/results/training.png?raw=true)

2. Evaluation

![Evaluation accuracies.](https://github.com/acoursey3/transformers-project/blob/main/results/accuracy.png?raw=true)

## Code Demo

See the file "demo.ipynb" for the code demo.

## Model Card
The model card is stored in the [modelcard.md](modelcard.md) file. Please carefully read this model card before attempting to use our pretrained models.

## Critical Analysis
- This project reveals that Transformer models may have advantages over RNN models even with limited data. This can impact model design decisions in future projects.
- We wrote code for Tree-Structured Parzen Estimator hyperparameter optimization, but computation time was an issue. The hyperparameters may not be optimal, so our results might change with optimal hyperparameters.
- We did not perform any gender/accent bias analysis
- Limited evaluation metrics
- This dataset may not be representative of their performance on other tasks, or even up-to-date with current Text-to-Speech algorithms

**Next Steps**
- Optimize hyperparameters
- Explore more datasets (different domains and tasks)
- Calculate more evaluation metrics

## Resource Links
- [FOR paper](https://bil.eecs.yorku.ca/wp-content/uploads/2020/01/FoR-Dataset_RR_VT_final.pdf)
- [FOR download link](https://bil.eecs.yorku.ca/datasets/#:~:text=scroll%20to%20access\)-,The%20Fake-or-Real%20Dataset,classifiers%20to%20detect%20synthetic%20speech.)
- [LSTM overview](https://medium.com/@ottaviocalzone/an-intuitive-explanation-of-lstm-a035eb6ab42c#:~:text=The%20LSTM%20architecture%20contrasts%20the,depends%20on%20the%20cell%20state.)
- [Paper comparing transformer and lstm for audio](https://arxiv.org/abs/1909.06317)
- [TPE hyperparameter optimization library](https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.TPESampler.html)
- [Hackers clone CEO's voice](https://www.wsj.com/articles/fraudsters-use-ai-to-mimic-ceos-voice-in-unusual-cybercrime-case-11567157402)
- [wav2vec Paper](https://arxiv.org/pdf/1904.05862.pdf)
