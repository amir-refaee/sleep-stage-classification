# sleep-stage-classification

Architecture written in PyTorch for model presented in [Sridhar et al.](https://www.nature.com/articles/s41746-020-0291-x) for sleep stage classification. 


# example usage

```python
net = SleepClassifier()

inpt = torch.rand(1200, 1, 256) # (n_sleep_epochs, n_channels, n_epoch_features)

yhat = net(inpt)  # output shape: [1, 4, 1200] corresponding to the logits for 4 sleeps classes
```

The dataset used in the paper is comprised of multiple sleep sessions. Each session is padded to a length of `(1, 7200)` where then a sliding window corresponding to a sleep epoch of 30-s reshapes the data into `(1200, 256)` associated with labels of shape `(1,1200)` classifiying each of the 30-s epochs in one of the sleep classes `[0, 1, 2, 3]` and `-1` which corresponds to padded data which should be ignored in training and evaluation.
