# Autograd:automatic differentiation(automatic backpropagation) 
# Central to all neural networks in pytorch, autograd package provides 
# automatic differentiation for all operations on tensors.

# autograd.Variable is the central class of package, it warps a tensor 
# and supports nearly all of operations on it. It contains three class:data,grad and creator
import torch
from torch.autograd import Variable
x = Variable(torch.ones(2,2), requires_grad = True)
print(x)

y = x+2
print(y) 
print(y.grad_fn) #y is created by operation, so have .grad_fn 

z = y*y*3
out = z.mean()
print(z,out)

# calculate gradient d(out)/dx as z = 3(x+2)^2
out.backward() # do backward for out(x)
print(x.grad) # get derivative(gradient) along x 

x = torch.randn(3)
x = Variable(x,requires_grad=True)
y = x*2
while y.data.norm()<1000:
    y = y*2
    
print(y)

gradients = torch.FloatTensor([0.1,1.0,0.0001])
y.backward(gradients)

print(x.grad)
print(gradients)

# help(Variable.grad)
