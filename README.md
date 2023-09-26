# Numerical methods for optimization, differentiation, integration and root-finding with applications in economic modelling

This repository contains the experiments and exercises with various numerical methods for root-finding, optimization, differentiation and integration. Some of these methods are applied to solving mainstream computational problems in economics, such as optimizing non-linear least squares and obtaining the asymptotic variance of estimators via differentiation. The functions and methods can be easily adapted for various applications in economic and financial modelling. More details of each file can be found in the comments.

## File details

[`differentiate_comparison.py`](differentiate_comparison.py): this Python script compares the Hessian matrix of non-linear least squares estimator (taking expectation of Hessian gives the Fisher Information) computed from numerical methods vs that derived analytically. It requires the ctb_sample data.

[`differentiate_simple.py`](differentiate_simple.py): this Python script is a demonstration of how simple numerical differentiation can be done by taking forward difference or central difference.

[`integrate_comparison.py`](integrate_comparison.py): this Python script compares the efficiency and accuracy of the outcomes of various numerical integration methods. Integration is used for taking expectation over random shocks in the intertemporal decision-making model under interest rate uncertainty.

[`numerical_integration.py`](numerical_integration.py): this Python script implements various numerical integration methods from scratch and compares their computation accuracies.

[`malthusian_model.py`](malthusian_model.py): this Python script implements the canonical Malthusian model in long-run growth of macroeconomics. Initial values and parameters can be given. Steady states and transition dynamics are computed.

[`optimize_nonlinear_least_squares.py`](optimize_nonlinear_least_squares.py): this Python script implements a full pipeline for optimizing a non-linear least squares objective via both derivative-based and derivative-free methods. It also runs robustness checks of the results.

[`optimize_simple.py`](optimize_simple.py): this Python script tests out numerical optimization methods from scipy.optimize module.

[`sentiment_prediction.py`](sentiment_prediction.py): this Python script loads the Kiva loan dataset and implements LASSO, Ridge regression and random forest to predict the numbers of days taken to get fully funded. It runs sentiment analysis and uses sentiment score as a covariate. 

[`solve_nonlinear_cournot.py`](solve_nonlinear_cournot.py): this Python script solves the classical game theory non-linear Cournot model using various numerical methods, by finding roots to a system of non-linear equations.

[`solve_nonlinear.py`](solve_nonlinear.py): this Python script demonstrates fixed point iteration from scratch to find roots to non-linear equations.

[`ctb_sample.dta`](ctb_sample.dta): the ctb sample data in .dta format.

[`kiva_loans_sample.csv`](kiva_loans_sample.csv): the Kiva loans dataset.

[`kiva_data_dict.xlsx`](kiva_data_dict.xlsx): data dictionary for the Kiva loans dataset.
