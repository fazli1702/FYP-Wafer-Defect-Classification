# Model Architecture

| Hyperparameter | Binary Classification | Multiclass Classification |
| :---- | :---- | :---- |
| Input Layer Shape (`in_features`) | Number of features | Same |
| Hidden Layer(s) | Problem specific, min:1, max:unlimited | Same |
| Neurons / hidden layer | Problem specific, usually 10-512 | Same |
| Output Layer Shape (`out_features`) | 1 | 1 per class |
| Hidden Layer Activation | ReLU (or others) | Same |
| Output Activation | Sigmoid | Softmax |
| Loss Function | Binary CrossEntropy| Cross Entropy |
| Optimizer | SGD (stochastic gradient descent), Adam | Same |


```python
# create a model
model = nn.Sequential(
    nn.Linear(in_features=3, out_features=100),     # input layer
    nn.Linear(in_features=100, out_features=100),   # hidden layer
    nn.ReLU()
    nn.Linear(in_features=100, out_features=3)      # output layer
)

# setup loss function + optimizer
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(params=model.parameters(), lr=0.001)

# training code
...

# testing code
...
```