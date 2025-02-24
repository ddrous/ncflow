<div style="text-align:center"><img src="docs/assets/Tax.png" /></div>

# Neural Context Flow
Neural Context Flow: A library for Generalizable Neural ODEs

## Instructions
1. Install the package: `pip install -e .`
2. Update the data generation script in `dataset.py`, or directly download the data from [Gen-Dynamics](https://anonymous.4open.science/r/gen-dynamics/)
3. Then set its hyperparameters and run the main script: `python main.py`

NCflow is built around 5 extensible modules: 
- a DataLoader: to store the dataset
- a Learner: a model and loss function
- A Trainer: to train
- a VisualTester: to test and visualize the results


## ToDo:
- [ ] Put the looses images here: Point to the Torch repo.
- [ ] Add the BibTex citation
- [ ] Open-source the weights and biases
- [ ] Add recommendation to run with nohup


If you like our work, please cite the corresponding paper: