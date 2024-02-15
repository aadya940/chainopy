---
title: 'ChainoPy: A Python Library for Discrete Time Markov Chains and Markov Chain Neural Networks'
tags:
    - Markov Chains
    - Markov Chain Neural Networks
    - Stochastic Analysis
    - High Performance Computing
authos:
    - name: Aadya A. Chinubhai
    - affiliation: 1
affiliations:
    - name: School of Engineering and Applied Sciences (SEAS), Ahmedabad University, Ahmedabad, Gujarat, India
      index: 1
bibliography: paper.bib
---

### Statement of Need
There are significant limitations in current Markov Chain packages that rely solely on pure NumPy and Python for implementation. Markov Chains often require iterative convergence-based algorithms, where Python's dynamic typing, Global Interpreter Lock (GIL), and garbage collection can hinder potential performance improvements like Parallelism. To address these issues, we enhance our library with extensions like Cython and Numba for efficient algorithm implementation. Additionally, we introduce a Markov Chain Neural Network [@awiszus2018markov] that simulates given Markov Chains while preserving statistical properties from the training data. This approach eliminates the need for post-processing steps such as sampling from the outcome distribution.

### Implementation

We implement two public classes `MarkovChain` and `MarkovChainNeuralNetwork` that contain core functionalities of the package. Performance itensive functions for the `MarkovChain` class are implemented in the `_backend` directory where a
custom cython backend is implemented circumventing drawbacks of python like the GIL, dynamic typing etc. The `MarkovChain` class implements various functionalities for discrete-time Markov chains. It provides methods for fitting the transition matrix from data, simulating the chain, calculating properties such as ergodicity, irreducibility, symmetry, and periodicity, as well as computing stationary distributions, absorption probabilities, expected time to absorption, and expected number of visits. It also supports visualization of the transition matrix and chain.  

We do the following key optimizations: 

- Efficient matrix power: If the matrix is diagonalizable, a Eigenvalue decomposition based Matrix power is Performed.

```math
[ A^n = V \Lambda^n V^{-1} ]
```

Where:

```
- \( A \) is the eigendecomposable matrix,
- \( V \) is the matrix of eigenvectors of \( A \),
- \( \Lambda \) is the diagonal matrix of eigenvalues of \( A \),
- \( n \) is the exponent for matrix power calculation.
```


- Parallel Execution: Some functions are parallelized (eg: `MarkovChain().is_absorbing()`)
- JIT compilation with Numba: Numba is used for just-in-time compilation to improve performance.
- `__slots__` usage: `__slots__` is used instead of `__dict__` for storing object attributes, reducing memory overhead.
- Caching decorator: Class methods are decorated with caching to avoid recomputation of unnecessary results.
- Direct LAPACK use: LAPACK function `dgeev` is directly used to calculate stationary-distribution via SciPy's 
`cython_lapack` API 
- Utility functions for visualization: Utility functions are implemented for visualizing the Markov chain.
- Sparse storage of transition matrix: The model is stored as a JSON object, and if 40% or more elements of the transition matrix are near zero, it is stored in a sparse format.

The `MarkovChainNeuralNetwork` implementation defines a neural network model, MarkovChainNeuralNetwork, using PyTorch for simulating Markov chain behavior. It takes a Markov chain object and the number of layers as input, with each layer being a linear layer. The model's forward method computes the output probabilities for the next state. The model is trained using stochastic gradient descent (SGD) with a learning rate scheduler. Finally, the model's performance is evaluated using the KL divergence between the original Markov chain's transition probabilities and those estimated from the simulated walks.

The steps to generate training data as described in [@awiszus2018markov]  are as follows:
    1. Input Data Augmentation: Add a random value (r) between 0 and 1 to the input data. This value influences the output, simulating the Markov chain's probabilistic nature.

    2. Cumulative Frequency Calculation: Calculate the cumulative frequency for each possible transition from the current state to the next states based on transition probabilities.

    3. Training Data Generation: Generate training data by sampling random numbers (r) and selecting the next state based on the calculated cumulative frequencies. This reflects the Markov chain's transition probabilities.

    4. Example: If the transition probabilities from state 1 to states 2, 3, and 4 are 1/3 each, the cumulative frequencies would be [0, 1/3, 2/3, 1]. For instance, with a random number of 0.5, the next state might be 3, resulting in the pair (0.5, 1, 0, 0, 0) → (0, 0, 1, 0) for state 1.

API of the library:

    - chainopy.MarkovChain(transition-matrix: ndarray, states: list)
            
            Public Methods
            -------------- 
            
            - fit(data, epsilon=1e-16)
            - simulate(initial_state, n_steps)
            - predict(initial_state)
            - adjacency_matrix()
            - nstep_distribution(n_steps)
            - is_ergodic()
            - is_symmetric()
            - stationary_dist()
            - is_absorbing()
            - is_aperiodic()
            - period()
            - is_irreducible()
            - is_transient(state)
            - is_recurrent(state)
            - fundamental_matrix()
            - absorption_probabilities()
            - expected_time_to_absorption()
            - expected_number_of_visits()
            - expected_hitting_time(state)
            - visualize_transition_matrix()
            - visualize_chain()
            - save_model(filename, epsilon=1e-16)
            - load_model(path)

    - chainopy.MarkovChainNeuralNetwork(chainopy.MarkovChain, num_layers)

            Public Methods
            --------------

            - train_model(num_samples, epochs, learning_rate, momentum=0.9, verbose=True, patience=500, factor=0.5)
            - get_weights()
            - simulate_random_walk(start_state, steps)
    
    - chainopy.divergance_analysis(MarkovChain, MarkovChainNeuralNetwork)

### Documentation, Testing and Benchmarking

For Documentation we use Sphinx. For Testing and Benchmarking the `MarkovChain` class we use the Pytest and PyDTMC [@pydtmc] package. 

The results are as follows:

- `is_absorbing` and `stationary_dist \ pi` Methods

| Transition-Matrix Size | 10            | 50            | 100           | 500           | 1000          | 2500          |
|------------------------ |---------------|---------------|---------------|---------------|---------------|---------------|
|                           | Mean                   | St. dev       | Mean          | St. dev       | Mean          | St. dev       | Mean          | St. dev       | Mean          | St. dev       | Mean          | St. dev       |
| Function                |               |               |               |               |               |               |
| 1. is_absorbing (ChainoPy) | 97.3ns        | 2.46ns        | 91.8ns        | 0.329ns       | 98ns          | 0.4ns         | 97.6ns        | 0.475ns       | 106ns         | 1.48ns        | 103ns         | 1.37ns        |
| 1. is_absorbing (PyDTMC)  | 386ns         | 5.79ns        | 402ns         | 2.01ns        | 417ns         | 3ns           | 416ns         | 2.44ns        | 418ns         | 0.837ns       | 433ns         | 6.3ns         |
| 2. stationary_dist (ChainoPy) | 1.47us     | 1.36us        | 93.4ns        | 5.26ns        | 96.6ns        | 3.9ns         | 550ns         | 344ns         | 753ns         | 685ns         | 857ns         | 850ns         |
| 2. pi (PyDTMC)            | 137us         | 12.9us        | 395ns         | 15.4ns        | 398ns         | 10.5ns        | 1.28us        | 1.79us        | 1.21us        | 1.71us        | 1.41us        | 1.85us        |



- `fit` vs `fit_sequence` Method:


| Number of Words          | 10            | 50            | 100           | 500           | 1000          | 2500          |
|--------------------------|---------------|---------------|---------------|---------------|---------------|---------------|
|                          |    Mean                     | St. dev       | Mean          | St. dev       | Mean          | St. dev       | Mean          | St. dev       | Mean          | St. dev       | Mean          | St. dev       |
| Function                 |               |               |               |               |               |               |
| 1. fit (ChainoPy)           | 116 µs        | 5.28 µs       | 266 µs        | 15 µs         | 496 µs        | 47.3 µs       | 6.58 ms       | 403 µs        | 23.6 ms       | 1.75 ms       | 587 ms        | 30.7 ms       |
| 1. fit_sequence (PyDTMC)    | 14 ms         | 1.74 ms       | 14.4 ms       | 1.17 ms       | 17.3 ms       | 2.18 ms       | 63.6 ms       | 6.63 ms       | 224 ms        | 5.84 ms       | 5.3 s         | 212 ms        |


- `simulate` Method

| Transition-Matrix Size | N-Steps | ChainoPy Mean | ChainoPy St. dev | PyDTMC Mean | PyDTMC St. dev |
|------------------------|---------|---------------|------------------|-------------|----------------|
| 10                     | 1000    | 22.8 ms       | 2.32 ms          | 28.2 ms     | 933 µs         |
|                        | 5000    | 86.8 ms       | 2.76 ms          | 155 ms      | 5.25 ms        |
| 50                     | 1000    | 17.6 ms       | 1.2 ms           | 29.9 ms     | 1.09 ms        |
|                        | 5000    | 84.5 ms       | 4.84 ms          | 161 ms      | 7.62 ms        |
| 100                    | 1000    | 21.6 ms       | 901 µs           | 37.4 ms     | 3.99 ms        |
|                        | 5000    | 110 ms        | 11.3 ms          | 162 ms      | 5.75 ms        |
| 500                    | 1000    | 24 ms         | 3.73 ms          | 39.6 ms     | 6.07 ms        |
|                        | 5000    | 112 ms        | 6.63 ms          | 178 ms      | 26.5 ms        |
| 1000                   | 1000    | 26.1 ms       | 620 µs           | 46.1 ms     | 6.47 ms        |
|                        | 5000    | 136 ms        | 2.49 ms          | 188 ms      | 2.43 ms        |
| 2500                   | 1000    | 42 ms         | 3.77 ms          | 59.6 ms     | 2.29 ms        |
|                        | 5000    | 209 ms        | 16.4 ms          | 285 ms      | 27.6ms         |

Apart from this, we test the `MarkovChainNeuralNetworks` by training them and comparing random walks between
the original `MarkovChain` (Right) object and those generated by `MarkovChainNeuralNetworks` (Left) through a Histogram.

The results for a 2 x 2 Markov Chain are as follows:

![2 x 2 Simulation][https://github.com/aadya940/chainopy/blob/master/figs/Simulation-MCNN-2x2.png]

The results for a 3 x 3 Markov Chain are as follows:

![3 x 3 Simulation][https://github.com/aadya940/chainopy/blob/master/figs/Simulation-MCNN-3x3.png]

## Conclusion

In conclusion, ChainoPy offers a Python library for discrete-time Markov Chains and includes features for Markov Chain Neural Networks, providing a useful tool for researchers and practitioners in stochastic analysis with efficient 
performance.


## References