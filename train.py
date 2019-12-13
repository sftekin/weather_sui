
def trainer(data_loaders, **kwargs):
    num_epoch = kwargs['num_epoch']

    for epoch in range(num_epoch):

        train(model, data_loaders['train'])

# def train(model, da)
