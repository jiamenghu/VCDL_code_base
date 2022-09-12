import torch 


a=torch.tensor([[1,2,3],[4,5,6]])
print(a)
b=torch.tensor([[-1,-2,-3],[-4,-5,-6]])
print(b)
print(torch.cat([a,b],1))

c=a.view(2,3,1)
d=b.view(2,3,1)

print(torch.cat([c,d],2).view(2,6))
