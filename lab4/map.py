import torch
# 9 predictions for 20 gts, the first and the last four are TP
# [1,0,0,0,0,1,1,1,1]
gt = 17
predictions = torch.hstack((torch.ones(1), torch.zeros(1), torch.ones(1)))
a = predictions.cumsum(0).div(torch.arange(len(predictions))+1)
b = torch.hstack((torch.ones(1),a,torch.zeros(1)))
c = predictions.cumsum(0).div(gt)
d = torch.hstack((torch.zeros(1),c,torch.ones(1)))

for i in range(len(b)-2, -1, -1):    
    b[i] = torch.max(b[i], b[i+1])
    
inds = torch.where(d[1:]!=d[:-1])[0]
ap = (b[inds+1]*(d[inds+1]-d[inds])).sum().numpy()
print("AP = {0:.4f}".format(ap))
