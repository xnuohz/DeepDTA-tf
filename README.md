# DeepDTA-tf
Tensorflow implementation of "DeepDTA deep drug-target binding affinity prediction"
## envs
* python 3.6
* tensorflow 1.12.0
## data
* kiba: 10 cross validation
* newkd: all for training
* drugbank: new drug, new target, new pair 5 cross validation
* FDA, Merck, XJ, YS, ZDC: just for prediction
## data format
* cross validation: ligands.csv, proteins.csv, inter.csv
* all training: ligands.csv, proteins.csv, pairs.csv
## benchmark
```txt
kiba as training set, predict new data(FDA & Merck) and calc AUC
```
## Config files
* data_xxx.cfg
```txt
[model]
data_path = path of training data
path = path for saving the model

[cv]
cv_num = 5
# 1: new pair, 2: new target, 3: new drug
problem_type = 1
```
* model.cfg
```txt
[model]
params...
```
* predict.cfg
```txt
Notice: change model.path & data.prediction for prediction
[model]
params...
path = ../../model/newkd

[data]
path = ../../data/
prediction = FDA,Merck,XJ,YS,ZDC
```
## How to run
* training
```bash
cd src/deepdta or cd/src/deepdti
# cross validation
python cv.py ../../config/model.cfg ../../config/data_kiba.cfg
# all for training
python main.py ../../config/model.cfg ../../config/data_newkd.cfg
# special conditions for training
python special.py ../../config/model.cfg ../../config/data_drugbank.cfg
```
* prediction
```bash
cd src/deepdta or cd/src/deepdti
python predict.py ../../config/predict.cfg
```
* DUD-E is different from other datasets, so I wrote the file separately.
```bash
# cross validation
python cv_dude.py ../../config/model.cfg ../../config/data_dude.cfg
# predict dude using trained DrugBank model
python predict_dude.py ../../config/predict.cfg
```
## Model file

| path | description |
| :-: | :-: |
| drugbank-newdrug | new drug 5 cv |
| drugbank-newtarget | new target 5 cv |
| drugbank-newpair | new pair 5 cv |
| drugbank-ecfp | ecfp + seq all training |
| drugbank-smiles | smiles + seq all training |
| kiba | smiles + seq all training & 10 cv |
| newkd | smiles + seq all training |

## Results
* AUCs/AUPRs(smiles + seq)

| dataset used in training | FDA | Merck | FDA + Merck |
| :-: | :-: | :-: | :-: |
| kiba | 0.5220/0.0984 | 0.3677/0.3513 | 0.3722/0.0801 |
| newkd | 0.5149/0.0481 | 0.3396/0.1494 | 0.6389/0.1108 |
| DB201707 | 0.5257/0.0413 | 0.7510/0.3879 | 0.5720/0.0643 |
| DB201707-del-P11388 | 0.6015/0.0922 | 0.5323/0.1938 | 0.5462/0.0880 |
| DUD-E | 0.5637/0.0510 | 0.6698/0.3256 | 0.6068/0.0828 |

* AUCs(fingerprints + seq)

| dataset used in training | FDA | Merck | FDA + Merck
| :-: | :-: | :-: | :-: |
| DB201707 | 0.3585 | 0.4719 | 0.4324 |

## DrugBank 201707 Result
1. CNN (smiles + seq)
* AUC 5 cross validation

| type | 1 | 2 | 3 | 4 | 5 |
| :-: | :-: | :-: | :-: | :-: | :-: |
| new target | 0.798 | 0.842 | 0.801 | 0.786 | 0.801 |
| new pair | 0.904 | 0.900 | 0.900 | 0.895 | 0.897 |
| new drug | 0.844 | 0.810 | 0.787 | 0.802 | 0.831 |

* AUPR 5 cross validation

| type | 1 | 2 | 3 | 4 | 5 |
| :-: | :-: | :-: | :-: | :-: | :-: |
| new target | 0.461 | 0.507 | 0.483 | 0.381 | 0.412 |
| new pair | 0.663 | 0.635 | 0.644 | 0.628 | 0.651 |
| new drug | 0.501 | 0.479 | 0.405 | 0.421 | 0.409 |

2. CNN (smiles + seq) vs ECFPCNN (fingerprints + seq)
* AUC 5 cross validation

| methods | new target | new drug | new pair |
| :-: | :-: | :-: | :-: |
| CNN | 0.844 | 0.872 | 0.924 |
| ECFPCNN | 0.837 | 0.857 | 0.912 |

* AUPR 5 cross validation

| methods | new target | new drug | new pair |
| :-: | :-: | :-: | :-: |
| CNN | 0.488 | 0.563 | 0.720 |
| ECFPCNN | 0.524 | 0.551 | 0.702 |

## Drop P11388 col, other as training, P11388 col as test

P11388 col: 539

| model | AUC | AUPR |
| :-: | :-: | :-: |
| CNN | 0.992 | 0.659 |
| ECFPCNN | 0.827 | 0.047 

DUD-E as training, P11388 col as test

| model | AUC | AUPR |
| :-: | :-: | :-: |
| CNN | 0.490 | 0.015 |

## DUD-E Result
* AUC 5 cross validation

| type | 1 | 2 | 3 | 4 | 5 |
| :-: | :-: | :-: | :-: | :-: | :-: |
| new target | 0.917 | 0.905 | 0.892 | 0.896 | 0.917 |

* AUPR 5 cross validation

| type | 1 | 2 | 3 | 4 | 5 |
| :-: | :-: | :-: | :-: | :-: | :-: |
| new target | 0.569 | 0.502 | 0.457 | 0.495 | 0.554 |

## NeoData Result
1. (708, 1512)
* AUC 10 cross validation

| type | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| new target | 0.897 | 0.895 | 0.873 | 0.916 | 0.899 | 0.835 | 0.912 | 0.918 | 0.786 | 0.874 |
| new pair | 0.927 | 0.929 | 0.920 | 0.951 | 0.949 | 0.944 | 0.950 | 0.936 | 0.936 | 0.949 |
| new drug | 0.894 | 0.880 | 0.873 | 0.874 | 0.880 | 0.902 | 0.879 | 0.916 | 0.886 | 0.894 |

* AUPR 10 cross validation

| type | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| new target | 0.676 | 0.677 | 0.597 | 0.658 | 0.686 | 0.373 | 0.707 | 0.713 | 0.195 | 0.521 |
| new pair | 0.721 | 0.744 | 0.724 | 0.756 | 0.729 | 0.724 | 0.769 | 0.763 | 0.719 | 0.774 |
| new drug | 0.707 | 0.639 | 0.506 | 0.559 | 0.525 | 0.670 | 0.560 | 0.687 | 0.618 | 0.703 |

* AUC 5 cross validation

| type | 1 | 2 | 3 | 4 | 5 |
| :-: | :-: | :-: | :-: | :-: | :-: |
| new target | 0.880 | 0.858 | 0.894 | 0.910 | 0.815 |
| new pair | 0.921 | 0.924 | 0.944 | 0.941 | 0.925 |
| new drug | 0.888 | 0.870 | 0.880 | 0.902 | 0.888 |

* AUPR 5 cross validation

| type | 1 | 2 | 3 | 4 | 5 |
| :-: | :-: | :-: | :-: | :-: | :-: |
| new target | 0.632 | 0.591 | 0.609 | 0.633 | 0.355 |
| new pair | 0.719 | 0.702 | 0.751 | 0.731 | 0.696 |
| new drug | 0.622 | 0.505 | 0.543 | 0.560 | 0.648 |

2. (549, 424) del none rows & cols
* AUC 10 cross validation

| type | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| new target | 0.862 | 0.862 | 0.800 | 0.877 | 0.796 | 0.845 | 0.852 | 0.865 | 0.797 | 0.849 |
| new pair | 0.891 | 0.885 | 0.902 | 0.923 | 0.927 | 0.915 | 0.925 | 0.908 | 0.909 | 0.914 |
| new drug | 0.753 | 0.807 | 0.804 | 0.893 | 0.729 | 0.763 | 0.822 | 0.805 | 0.820 | 0.854 |

* AUPR 10 cross validation

| type | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| new target | 0.580 | 0.527 | 0.502 | 0.607 | 0.468 | 0.439 | 0.496 | 0.529 | 0.417 | 0.478 |
| new pair | 0.667 | 0.699 | 0.651 | 0.663 | 0.715 | 0.684 | 0.725 | 0.697 | 0.667 | 0.714 |
| new drug | 0.385 | 0.530 | 0.444 | 0.700 | 0.362 | 0.436 | 0.486 | 0.433 | 0.519 | 0.403 |

* AUC 5 cross validation

| type | 1 | 2 | 3 | 4 | 5 |
| :-: | :-: | :-: | :-: | :-: | :-: |
| new target | 0.837 | 0.826 | 0.815 | 0.824 | 0.775 |
| new pair | 0.874 | 0.890 | 0.913 | 0.900 | 0.894 |
| new drug | 0.734 | 0.833 | 0.800 | 0.804 | 0.780 |

* AUPR 5 cross validation

| type | 1 | 2 | 3 | 4 | 5 |
| :-: | :-: | :-: | :-: | :-: | :-: |
| new target | 0.504 | 0.605 | 0.449 | 0.485 | 0.349 |
| new pair | 0.653 | 0.680 | 0.676 | 0.654 | 0.659 |
| new drug | 0.386 | 0.524 | 0.412 | 0.446 | 0.326 |

## ToDo List:

folds_neo_708_1512: (708, 1512) 5 cv split
folds: (549, 424) 5 cv split

(549, 424): DeepDTA & NeoDTI 5 cv in new d, t, p

neo t
neo d
neo p