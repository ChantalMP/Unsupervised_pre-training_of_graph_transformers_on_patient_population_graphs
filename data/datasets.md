## Datasets

we use two datasets:

- TADPOLE:
    - https://tadpole.grand-challenge.org/Data/
    - we use only the baseline visits of all patients

- MIMIC-III:
    - we use the preprocessed dataset published by McDermott et al in "A Comprehensive EHR Timeseries Pre-training Benchmark" [1]

## Data Directory Structure:

```
data
│
└───mimic-iii-0
│   └─── drop
│   │	 └─── <splits to train with limited labels generated with graphormer.utils.utils.generate_labels_to_drop_mimic>
│   └─── rotations
│   │	 └─── train_patients_rot_0.json
│	│	 └─── val_patients_rot_0.json
│	│	 └─── test_patients_rot_0.json
│	│	 └─── ... for all rotations / folds
│   └─── train
│    	 └─── patient_<id_i>
│    	 │	  └─── <csv files for all patient data of patient with id_i for training set of rotation 0>
│    	 │	  └─── processed
│		 │		   └─── <processed graph data will be saved here>
│		 │     └─── raw
│		 │		   └─── rot0
│		 │		   		└─── random_graph_subset_0.json
│		 │		   		└─── ... for all subset graphs of rotation 0 (randomly split patients to required graph size)
│		 │		   └─── ...
│    	 └─── <splits to train with limited labels generated with graphormer.utils.utils.generate_labels_to_drop_balanced_tadpole>
		 val
│    	 └─── cross_val
│    	 │	  └─── <cross-val folds generated with graphormer.utils.utils.generate_stratified_cross_val_split>
│    	 └─── <splits to train with limited labels generated with graphormer.utils.utils.generate_labels_to_drop_balanced_tadpole>
		 test
│    	 └─── cross_val
│    	 │	  └─── <cross-val folds generated with graphormer.utils.utils.generate_stratified_cross_val_split>
│    	 └─── <splits to train with limited labels generated with graphormer.utils.utils.generate_labels_to_drop_balanced_tadpole>
│   
└───tadpole
    └─── processed
    │	 └─── <processed graph data will be saved here>
    └─── raw
    │	 └─── tadpole_numerical.csv (tadpole data)
    └─── split
    	 └─── cross_val
    	 	  └─── <cross-val folds generated with graphormer.utils.utils.generate_stratified_cross_val_split>
    	 └─── <splits to train with limited labels generated with graphormer.utils.utils.generate_labels_to_drop_balanced_tadpole>
└───sepsis
    └─── processed
    │	 └─── <processed graph data will be saved here>
    └─── raw
    │	 └─── cross_val_A
              └─── <cross-val folds generated with TODO (dataset_graph_split_sepsis.py) for dataset A>
    │	 └─── cross_val_B
              └─── <cross-val folds generated with TODO (dataset_graph_split_sepsis.py) for dataset B>
    └─── split
    	 └─── cross_val
    	 	  └─── <cross-val folds generated with graphormer.utils.utils.generate_stratified_cross_val_split>
    	 └─── <splits to train with limited labels generated with graphormer.utils.utils.generate_labels_to_drop_balanced_tadpole>

``` 

[1] McDermott, M., Nestor, B., Kim, E., Zhang, W., Goldenberg, A., Szolovits, P., Ghassemi, M.: A comprehensive ehr timeseries pre-training benchmark.
In: Proceedings of the Conference on Health, Inference, and Learning (2021)
