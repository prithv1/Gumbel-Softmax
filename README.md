# Gumbel-Softmax
## Description
A torch implementation of gumbel-softmax trick. Gumbel-Softmax is a continuous distribution on the simplex that can approximate
categorical samples, and whose parameter gradients can be easily computed via the reparameterization trick.
## Papers
[Categorical Reparameterization with Gumbel-Softmax](https://arxiv.org/pdf/1611.01144v2.pdf)
## Usage
Do the following :  
`require 'gum_softmax.lua'`  
`require 'nn'`  
`model = nn.Sequential()`  
`temperature = 1e-5`  
`model:add(nn.GumSoftMax(temperature))`  
