## Datasets

we use three datasets:

- TADPOLE:
    - https://tadpole.grand-challenge.org/Data/
    - we use only the baseline visits of all patients

- MIMIC-III:
  - we use the preprocessed dataset published by McDermott et al in "A Comprehensive EHR Timeseries Pre-training Benchmark" [1]

- Early prediction of Sepsis:
  - https://physionet.org/content/challenge-2019/1.0.0/
## Data Directory Structure:

```
data
│
└───mimic-iii-0
│   └─── drop
│   │	   └─── <splits to train with limited labels generated with graphormer.utils.utils.generate_labels_to_drop_mimic>
│   └─── rotations <we used rotations provided in pre-processed data of McDermott et al.: these files just contain list of patient ids>
│   │	   └─── train_patients_rot_0.json
│	  │	   └─── val_patients_rot_0.json
│	  │	   └─── test_patients_rot_0.json
│	  │	   └─── ... for all rotations / folds
│   └─── train
│   │ 	 └─── patient_<id_i>
│   │ 	 │	  └─── <csv files for all patient data of patient with id_i for training set of rotation 0>
│   │ 	 └─── processed
│	  │	 │	      └─── <processed graph data will be saved here>
│	  │	 └─── raw
│	  │	 	  └─── rot0
│	  │	 	       └─── random_graph_subset_0.json
│	  │	 	       └─── ... for all subset graphs of rotation 0 (randomly split patients to required graph size), generated with graphormer.data_preprocessing.graph_subsampling_mimic.create_subgraphs_mimic_random
│	  └─── val: same structure as train
│   └─── test: same structure as train
│   
└───tadpole
│   └─── processed
│   │	   └─── <processed graph data will be saved here>
│   └─── raw
│   │	   └─── tadpole_numerical.csv (tadpole data as csv file, one row per patient, one column per feature)
│   └─── split
│   	   └─── cross_val
│   	   │	  └─── <cross-val folds generated with graphormer.utils.utils.generate_stratified_cross_val_split>
│   	   └─── <splits to train with limited labels generated with graphormer.utils.utils.generate_labels_to_drop_balanced_tadpole>
└───sepsis
    └─── training_setA
    │    └─── <downloaded patient .psv files for training set A>
    └─── training_setB
    │    └─── <downloaded patient .psv files for training set B>
    └─── processed
    │	   └─── <processed graph data will be saved here>
    └───raw
    	  └─── cross_val_A
        │    └─── <cross-val folds generated with graphormer.data_preprocessing.dataset_graph_split_sepsis.create_cross_val_graphs for dataset A>
    	  └─── cross_val_B
             └─── <cross-val folds generated with graphormer.data_preprocessing.dataset_graph_split_sepsis.create_cross_val_graphs for dataset B>
``` 

[1] McDermott, M., Nestor, B., Kim, E., Zhang, W., Goldenberg, A., Szolovits, P., Ghassemi, M.: A comprehensive ehr timeseries pre-training benchmark.
In: Proceedings of the Conference on Health, Inference, and Learning (2021)
