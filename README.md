# ChainoPy 1.0
A Python 🐍 Package for Markov Chains and Markov Chain Neural Networks.

## Why ChainoPy?
- Covers most of the fundamental agorithms for Markov Chain Analysis
- Memory efficient Model saving 
- Faster than other libraries (eg: 5x Faster than PyDTMC)
- First Package to contain functions to build equivalent [Markov Chain Neural Networks](https://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w42/Awiszus_Markov_Chain_Neural_CVPR_2018_paper.pdf) from Markov Chains.
  


# How to Install ChainoPy?

Before you begin, ensure you have the following installed on your system:
- Python (>= 3.9 )

### 1. Clone the Repository
Clone the Chainopy repository to your local machine using Git:

```bash
git clone https://github.com/username/Chainopy.git
```

```bash
cd chainopy
```
### 2. Install Dependencies
```bash
pip install -r requirements.txt
pip install -r requirements-test.txt
```

### 3. Compile Cython Modules
```bash
python3 setup.py build_ext --inplace
```

### 4. Verify Installation
Run tests as described in the tests section

### 5. Install 
```bash
pip install .
```


# How to run ChainoPy Tests?
 1. Clone the project locally 
 2. Install packages mentioned in `requirements_test.txt`
 3. Travel to the directory containing `test_markovchain.py`
 4. Run the following command:
```bash
python -m pytest test_markovchain.py
```

You're all Set! 😃 👍


# The Basics
```{bash}
>>> mc = chainopy.MarkovChain([[0, 1], [1, 0]], states = ["Rain, "No-Rain"])
>>> neural_network = chainoy.MarkovChainNeuralNetwork(mc, num_layers = 5)
```
