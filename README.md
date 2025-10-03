# Biogeography
Modeling of Species

## EcoGP (Model)
The model is found in [EcoGP.py](models/EcoGP.py)

## Config
The [configs](configs/) folder contains the different configurations.

### Config Sharing
Should the data be stored in different places and one would like to use the same config file, then please add a variable with the path to the folder containing the data.
Currently, [data_folder_path.py](configs/data_folder_path.py) is containing the variable 'data_folder_path = "/path/to/data"' – should it not be created, please add the variable with your intended path.  

## Baselines
Baseline models can be found under the [models/baselines](models/baselines) folder.

## Datasets
The butterfly and central park datasets are in the [data](data/) folder.