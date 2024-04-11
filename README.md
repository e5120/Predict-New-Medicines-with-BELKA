# [Predict-New-Medicines-with-BELKA](https://www.kaggle.com/competitions/leash-BELKA/overview)

## Download Data

```
kaggle competitions download -c leash-BELKA
```

## Preprocess Data

```
> export PYTHONPATH=/path/to/repo
> python run/prepare_data.py -m stage=train,test
```

## Train

```
> export PYTHONPATH=/path/to/repo
> python run/train.py
```


## Inference

```
> export PYTHONPATH=/path/to/repo
> python run/inference.py
```
