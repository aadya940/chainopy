![logo1](https://github.com/aadya940/chainopy/assets/77720426/9c8d3781-945a-4ccb-a70f-2515cc1a8be6)

# ChainoPy 1.0
A Python 🐍 Package for Markov Chains, Markov Chain Neural Networks and Markov Switching Models.

## Why ChainoPy?
- Covers most of the fundamental agorithms for Markov Chain Analysis
- Memory efficient Model saving 
- Faster than other libraries (eg: 5x Faster than PyDTMC)
- First Package to contain functions to build equivalent [Markov Chain Neural Networks](https://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w42/Awiszus_Markov_Chain_Neural_CVPR_2018_paper.pdf) from Markov Chains.
- Contains Markov Switching Models for Univariate Time Series Analysis
  


# How to Install ChainoPy?

### Build from Source

Before you begin, ensure you have the following installed on your system:
- Python (>= 3.9 )

### 1. Clone the Repository
Fork and Clone the Chainopy repository to your local machine using Git:

```bash
git clone https://github.com/aadya940/chainopy.git
```

Navigate to the directory which contains the `pyproject.toml` file.

### 2. Install the package
```bash
python -m build
```


# How to run ChainoPy Tests?
 1. Clone the project locally 
 2. Install packages mentioned in `requirements.txt` and `requirements_test.txt`
 3. Navigate to the directory containing `tests` folder
 4. Run the following command:
```bash
python -m pytest tests/
```

You're all Set! 😃 👍


# The Basics
Create Markov Chains and Markov Chain Neural Networks as follows:
```{bash}
>>> import chainopy
>>> mc = chainopy.MarkovChain([[0, 1], [1, 0]], states = ["Rain", "No-Rain"])    # Creates a two-states Markov Chain stored in `mc`.
>>> neural_network = chainopy.MarkovChainNeuralNetwork(mc, num_layers = 5)    # Creates a 5-layered Neural Network that simulates `mc`. 
```

![image](https://github.com/aadya940/chainopy/blob/master/figs/Simulation-MCNN-2x2.png)

Create a Markov Switching Model as follows:

```{bash}
>>> import numpy as np
>>> import random
>>> from chainopy import MarkovSwitchingModel
>>> X = np.random.normal(0, 1, 1000) + np.random.logistic(5, 10, 1000) # Generate Random Training Data
>>> regime_col = [random.choice(["High", "Low", "Stagnant"]) for _ in range(1000)] # Generate Regimes for Training Data
>>> mod = MarkovSwitchingModel()
>>> mod.fit(X, regime_col)
>>> y, regime_y = mod.predict("High", steps=20)
```

Generates Data as follows:
- `X`: We generate 1000 data points by combining a normal distribution (mean = 0, standard deviation = 1) with a logistic 
distribution (mean = 5, scale = 10). This creates a complex dataset with variations.
- `regime_col`: We assign one of three possible regimes ("High", "Low", "Stagnant") to each data point. This is done by randomly
selecting one of these regimes for each of the 1000 data points.

Later, Creates a Markov Switching Model using `chainopy.MarkovSwitchingModel` with 3 regimes (High, Low and Stagnant) and 
predicts the next twenty steps if the start states is "High". 

### Example - Apple Weekly High Stock data prediction using chainopy.MarkovSwitchingModel
![image](https://github.com/aadya940/chainopy/assets/77720426/2d3ed6c0-5936-4fbe-9984-fdbe33e85e9a)

# How to Contribute?

1. Fork the Project.
2. Clone the Project locally.
3. Create a New Branch to Contribute.
4. run `pip install -r requirements.txt` and `pip install -r requirements_test.txt` to download dependencies.
5. Do the changes of interest (Make sure to write docstrings).
6. Write Unit Tests and test your implementation.
7. Format the code using the Black Formatter.
8. Push the changes and submit a Pull Request.

Note: If your implementation is Cython, justify its usage in your PR to make the code more maintainable.
