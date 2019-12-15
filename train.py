from models.conv_lstm import ConvLSTM


def trainer(data_loaders, **kwargs):
    num_epoch = kwargs['finetune_params']['epoch']
    batch_size = kwargs['batch_params']['batch_size']
    model = ConvLSTM(kwargs['constant_params'],
                     kwargs['finetune_params'])

    for epoch in range(num_epoch):
        print('-*-' * 12)
        print('Epoch: {}/{}'.format(epoch, num_epoch))
        print('-*-' * 12)

        model.reset_per_epoch(batch_size=batch_size)
        train_loss = _train(model, data_loaders['train'])
        val_loss = _evaluate(model, data_loaders['validation'])

        print('Training Loss: {:.2f}, '
              'Validation Loss {:.2f}'.format(train_loss, val_loss))

    return model

def _train(model, dataloader):
    batch_size = len(dataloader)
    running_loss = 0
    for grid, label_grid in dataloader:
        running_loss += model.fit(grid, label_grid)
    return running_loss / batch_size


def _evaluate(model, dataloader):
    batch_size = len(dataloader)
    running_loss = 0
    for grid, label_grid in dataloader:
        grid = grid.permute(0, 1, 4, 2, 3)
        label_grid = label_grid.permute(0, 1, 4, 2, 3).cpu().numpy()

        pred = model.predict(grid)
        running_loss += model.score(label_grid, pred)
    return running_loss / batch_size



