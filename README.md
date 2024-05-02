# [Predict-New-Medicines-with-BELKA](https://www.kaggle.com/competitions/leash-BELKA/overview)

## Computational Resources

- Intel(R) Core(TM) i5-8400 CPU @ 2.80GHz
- 64GB RAM
- 2x NVIDIA GeForce GTX 1080Ti

## Download Data

```
$ kaggle competitions download -c leash-BELKA
```

## Download this repository

```
$ https://github.com/e5120/Predict-New-Medicines-with-BELKA.git
$ cd Predict-New-Medicines-with-BELKA
$ export PYTHONPATH=.
```

## Prepare Data

```
$ python run/prepare_data.py phase=split  # It takes 20-30 minutes
$ python run/prepare_data.py phase=aggregate  # It takes a few hours
$ python run/prepare_data.py phase=cross_validation  # It takes a few minuts
```

## Preprocess Data

```
$ python run/prepare_data.py -m stage=train,test preprocessor=chemberta  # It takes a few hours
$ python run/prepare_data.py -m stage=train,test preprocessor=graph  # It takes 8-9 hours
```

## Train

```
$ python run/train.py dataset=lm_dataset model=lm_model exp_name=exp_chemberta
$ python run/train.py dataset=graph_dataset model=graph_model exp_name=exp_graph
```

## Inference

```
$ python run/inference.py --config-path /path/to/repo/output/exp_chemberta/single/.hydra
$ python run/inference.py --config-path /path/to/repo/output/exp_graph/single/.hydra
```
