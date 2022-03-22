import torch
import torchvision as tv


def mnist_dataloaders(mnist_path, batch_size=128, use_cuda=True, **inputs):
    '''Returns a train and test dataloader for mnist dataset'''
    trans = tv.transforms.ToTensor()
    print(inputs['data'])    
    if inputs['data']=='MNIST':
        train_set = tv.datasets.MNIST(root=mnist_path, train=True, transform=trans,
                                  download=inputs['download'])
        test_set = tv.datasets.MNIST(root=mnist_path, train=False, transform=trans)
    else:
        print(inputs['data']) 
        train_set = tv.datasets.FashionMNIST(root=mnist_path, train=True, transform=trans,
                                  download=inputs['download'])
        test_set = tv.datasets.FashionMNIST(root=mnist_path, train=False, transform=trans)
    dataloader_kwargs = {'num_workers': 1, 'pin_memory': use_cuda}
   
    train_loader = torch.utils.data.DataLoader(
        dataset=train_set, batch_size=batch_size, shuffle=True, **dataloader_kwargs
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_set, batch_size=batch_size, shuffle=False, **dataloader_kwargs)

    return train_loader, test_loader


def one_hot(vec, n_classes):
    '''
    Converts a vector in target indices to a matrix of 1 hots
    args:
        vec is a tensor
        n_classes is the number of classes in the one hot matrix
    '''
    v = torch.zeros(vec.shape[0], n_classes)
    for i, j in enumerate(vec):
        v[i, int(j)] = 1.
    return v


def categorical_accuracy(y_true, y_pred):
    y_pred = torch.max(y_pred, -1)
    y_pred = y_pred[1].float()
    return torch.mean(
        torch.eq(y_pred, y_true).float()
    )


model_configs = {
    'MNIST': {
        'n_conv_channel': 256,
        'n_primary_caps': 8,
        'primary_cap_size': 1152,
        'output_unit_size': 16,
        'n_routing_caps': 3
    }
}
