# Nodax
Inductive bias learning for dynamical systems

This is a one-stop shop for all things neural odes for dynamical systems.


Nodax is built around 5 extensible modules: 
- a DataLoader: to store the dataset
- a Learner: a model and loss function
- A Trainer: to train
- a VisualTester: to test and visualize the results
- a HPFactory: to find hyper-parameters for our models

Diagram showing the flow across the modules.


A few neural ODE implemented models:
- One-Per-Env
- One-For-All
- Context-Informed
- Neural Context Flow
