# Computational Mathematics Project 

This project has been developed by [Gabriele Pisciotta](https://github.com/GabrielePisciotta/) and [Marco Sorrenti](https://github.com/MarcoSorrenti) in the context of the Computational Mathematics course at the University of Pisa, held by professor Frangioni and professor Poloni. We also thank [Alessandro Cudazzo](https://github.com/alessandrocuda/) for the support and its inspiring code. 

For the full report, covering also the theoretical basis and the results obtained in this case study, [see here](./REPORT.pdf). 

Please note that the goal in this project is slighty different from the goals we consider in Machine Learning projects (i.e.: we're not interested in the generalization ability here, but we just want to find minima/maxima!)
 
# Content of the project

- Model: Neural Network (Multi-Layer Perceptron)

- Optimizers: Gradient Descent with Momentum, L-BFGS

- Loss function: (Regularized) Mean Squared Error

- Activation functions: linear, sigmoid, 

- Regularization: L2

- Utilities: rescaling, 1-of-K encoding, plot routines


# Guide

## Requirements installation
- Run `python3 -m pip install -r requirements.txt`

## Running

- Run `python3 main_monk{N}.py`, with {N} in {1,2,3} to replicate the experiments on the MONK 1, MONK 2, MONK 3 datasets
- Run `python3 main_cup.py` to run the experiments on the CUP dataset. 

## Results

The results will be outputted in the `results` folder generated, with a subfolder for each dataset. 

Here you will find:

- convergence rate plot (two optimizers in the same)
- convergence rate plots (one for each optimizer)
- losses plots (two in the same)
- losses plots (one for each optimizer)
- CSV results (for each dataset):
  - f* 
  - ||g|| 
  - number of iteration
  - total time
  - mean time per iteration
  - stop reason 
- idem, but in latex 
