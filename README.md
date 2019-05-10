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
* AUCs(smiles + seq)

| dataset used in training | FDA(old/new) | Merck(old/new) | FDA + Merck(old/new)
| :-: | :-: | :-: | :-: |
| kiba | 0.4549/ | 0.3698/ | 0.6055/ |
| newkd | 0.4136/ | 0.6500/ | 0.5914/ |
| DB201707 | 0.5257/0.5376 | 0.7510/0.6844 | 0.5720/0.5617 |
| DB201707-del-P11388 | 0.6015/0.5059 | 0.5323/0.5490 | 0.5462/0.4842 |

* AUCs(fingerprints + seq) only new

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
| ECFPCNN | 0.827 | 0.047 |

## Tips:
FDA & Merck: some wrong with calc raw smiles fingerprints,
so raw smiles file is renamed to ligands_raw.csv, and the updated file
is renamed as ligands_v2.csv which currently called ligands.csv.


