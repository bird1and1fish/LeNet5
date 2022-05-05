# LeNet5
realize LeNet5 in pytorch,
net.py is origin network,
net_quant.py is a network quantified on the basis of net,
you should first use train.py to train a model of net,
then use linear_quant function in net_quant.py to quantify the model,
finally you can use test_quant.py to Validate the quantized model.
