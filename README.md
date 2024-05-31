
# Linear-Reg From Scratch 

The Google Colab Notebook contains the Pytorch implementation of Linear Regression Algorithim from the Scratch.
The Model is trained purely on the basis linear model provided in Pytorch core tensors.



## Authors

- [@vishal351980](https://www.github.com/octokatherine)


## Usage/Examples
### The Traning Part of the Model
```python 
def fit(epochs,model,loss_fn,opt,train_dl):
  for epoch in range(epochs):
    # training in batches 
    for x_b,y_b in train_dl:
      pred = model(x_b)
      loss = loss_fn(pred,y_b)
      # computer the gradient 
      loss.backward()
      opt.step()
      opt.zero_grad()
    
    if (epoch+1)%10 ==0:
      print('Epoch [{}/{}], Loss: {:.4f}'.
      
      format(epoch+1, epochs, loss.item()))
# The prediction part of the model
preds = model(inputs)

## 🚀 About Me
I am a Machine learning , Deep learning and Computer Vision Enthusiast currently focusing on OpenCv for advanced Computer Vision and CNN's Architechure in Deep learning.

