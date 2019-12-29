import torch

from models.conv_lstm import ConvLSTM
from models.traj_gru import TrajGRU
from models.ema import EMA
from models.sma import SMA


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def trainer(batch_gens, **kwargs):
    num_epoch = kwargs['finetune_params']['epoch']
    batch_size = kwargs['batch_params']['batch_size']

    # model = ConvLSTM(kwargs['constant_params'],
    #                  kwargs['finetune_params'])
    # model = TrajGRU(kwargs['constant_params'],
    #                 kwargs['finetune_params'])
    # model = EMA(kwargs['constant_params'],
    #             kwargs['finetune_params'])
    model = SMA(kwargs['constant_params'],
                kwargs['finetune_params'])

    model = model.to(device)

    for epoch in range(num_epoch):
        print('-*-' * 12)
        print('Epoch: {}/{}'.format(epoch, num_epoch))
        print('-*-' * 12)

        # model.reset_per_epoch(batch_size=batch_size)
        train_loss = _train(model, batch_gens['train'])
        val_loss = _evaluate(model, batch_gens['validation'])

        print('Training Loss: {:.2f}, '
              'Validation Loss {:.2f}'.format(train_loss, val_loss))

    return model


def _train(model, batch_gen):
    running_loss = 0
    count = 0
    for grid, label_grid in batch_gen.batch_next():
        count += 1
        running_loss += model.fit(grid, label_grid, batch_idx=count)
    return running_loss / count


def _evaluate(model, batch_gen):
    running_loss = 0
    count = 0
    for grid, label_grid in batch_gen.batch_next():
        count += 1
        label_grid = label_grid.permute(0, 1, 4, 2, 3).cpu().numpy()

        pred = model.predict(grid)
        running_loss += model.score(label_grid, pred)
    return running_loss / count



