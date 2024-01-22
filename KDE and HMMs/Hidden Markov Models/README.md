# In-Task Questions

## 3.4.1 
You are Chottu, a casino inspector, whose job is to investigate whether a casino is trading out a fair die for a loaded (unfair) die and scamming the customers. At the same time, you are also well-versed in Statistical Methods in Artificial Intelligence, and you realize that you can use Hidden Markov Models to solve this case. You need not implement the HMM from scratch but are allowed to use any standard library. But please make sure you are familiar with the underlying theoretical concepts

---
### Question 3.4.2.4 : 
- What problem in Hidden Markov Models does this task correspond to?

*Solution*
- This part corresponds to the **Decoding** problem of the HMMs, where we are given HMM parameters and observations, we have to find the most probable sequence of hidden states.
- Algorithms:
    - Viterbi Algorithm: Finds the most probable sequence of hidden states by maximizing the joint probability of observations and states.
    - Posterior Decoding: Calculates the posterior probability of states given observations and then selects the states with the highest probability.

***Note : The estimation of best model also involves Evaluation Problem of the HMMs*** 

---
### Question 3.4.3.2 : 
- What problem in Hidden Markov Models does this task correspond to?

*Solution*
- This part corresponds to the **Learning** problem of the HMMs, where given a set of observations, we have to estimate the parameters (more particularly, transition probabilities) of the HMM.

- Algorithms:
    - Expectation-Maximization (EM) Algorithm: Used for parameter estimation in HMMs. The Baum-Welch algorithm is a specific instance of EM for HMMs.
    - Maximum Likelihood Estimation (MLE): An optimization approach to maximize the likelihood of the observed data with respect to the model parameters.
---

### Question 3.4.4.2 : 
- What problem in Hidden Markov Models does this task correspond to?

*Solution*
- This part corresponds to the **Learning** problem of the HMMs, where given a set of observations, we have to estimate the parameters (more particularly, emission probabilities) of the HMM.

- Algorithms:
    - Expectation-Maximization (EM) Algorithm: Used for parameter estimation in HMMs. The Baum-Welch algorithm is a specific instance of EM for HMMs.
    - Maximum Likelihood Estimation (MLE): An optimization approach to maximize the likelihood of the observed data with respect to the model parameters.
---

