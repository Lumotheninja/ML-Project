# HMM by hand
Implement Hidden Markov Models by hand with 4 POS datasets

| Question | Description                                                                                                                                                  |
|----------|--------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 2.1      | Write a function that estimates the emission parameters from the training set using MLE (maximum likelihood estimation)                                      |
| 2.2      | Set k to 1, implement add k smoothing into your function for computing the emission parameters.                                                              |
| 2.3      | Compare your outputs and the gold-standard outputs in dev.out and report the precision, recall and F scores of such a baseline system for each dataset.      |
| 3.1      | Write a function that estimates the transition parameters from the training set using MLE (maximum likelihood estimation)                                    |
| 3.2      | Use the estimated transition and emission parameters, implement the Viterbi algorithm taught in class to compute the following (for a sentence with n words) |
| 3.3      | Describe the Viterbi algorithm used for decoding such a second-order HMM model and implement it.                                                             |
| 5        | Now, based on the training and development set, think of a better design for developing improved NLP systems for tweets using any model you like.            |

# Design challenge
For the design challenge, we created a total of 3 models, first order HMM with JM smoothing, second order HMM with JM smoothing and third order HMM with JM smoothing. We used JM smoothing as there might be a lot of “holes” in the higher order HMM models as the probabilities get smaller, hence it might be a good idea to interpolate lower order models with higher order ones. It also ensures that unknown words can be handled in a smooth manner. After testing our models on the EN and FR dataset, we chose the first order HMM with JM smoothing as it was having the best results