<div style="text-align:center"><img src="docs/assets/Tax.png" /></div>

# NCFlow (previously called Nodax and TAX)
Neural Context Flow: A library for Generalisable Neural ODEs

This is a one-stop shop for all things neural odes for dynamical systems (hopefully, WIP).

What to do: Install the library and run the scripts in the folder `examples/`. 

`pip install -e .`


NCflow is built around 5 extensible modules: 
- a DataLoader: to store the dataset
- a Learner: a model and loss function
- A Trainer: to train
- a VisualTester: to test and visualize the results

A few neural ODE implemented models:
- One-Per-Env
- One-For-All
- Context-Informed
- Neural Context Flow
