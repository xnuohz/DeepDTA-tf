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
## Model file

| path | description |
| :-: | :-: |
| drugbank-newdrug | new drug 5 cv |
| drugbank-newtarget | new target 5 cv |
| drugbank-newpair | new pair 5 cv |
| drugbank-ecfp | ecfp + seq all training |
| drugbank-smiles | smiles + seq all training |
| kiba | 10 cv |
| newkd | smiles + seq all training |

## Results
* AUCs(smiles + seq)

| dataset used in training | FDA | Merck | FDA + Merck
| :-: | :-: | :-: | :-: |
| kiba | 0.4803 | 0.5365 | 0.5083 |
| newkd | 0.5250 | 0.6264 | 0.5551 |
| DB201707 | 0.5262 | 0.6302 | 0.5516 |
| DB201707-del-P11388 | 0.4808 | 0.5323 | 0.4929 |

## DrugBank 201707 Result
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

## Drop P11388 col, other as training, P11388 col as test

| model | AUC | AUPR |
| :-: | :-: | :-: |
| CNN | 0.885 | 0.121 |
| CNN-dropout | 0.980 | 0.565 |
| ECFPCNN | 0.888 | 0.084 |
| ECFPCNN-dropout | 0.5 | 0.508 |

## TODO List:
(After adding dropout)

* kiba training, FDA & Merck test
* newkd training, FDA & Merck test

all these are based on CNN method.