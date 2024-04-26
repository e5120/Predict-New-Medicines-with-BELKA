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
$ python run/prepare_data.py -m stage=train,test phase=split
$ python run/prepare_data.py phase=aggregate
```

## Preprocess Data

```
$ python run/prepare_data.py -m stage=train,test preprocessor=chemberta
```

## Train

```
$ python run/train.py dataset=lm_dataset model=lm_model exp_name=exp_chemberta
```

## Inference

```
$ python run/inference.py --config-path /path/to/repo/output/exp_chemberta/single/.hydra
```
