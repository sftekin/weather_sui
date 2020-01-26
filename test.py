import numpy as np
import matplotlib.pyplot as plt


def test(model, batch_gen):
    model.eval()

    heat_map = plt.figure('heatmap', figsize=(2.3 * 2, 2.3 * 5))
    time_plot = plt.figure('time_plot')
    selected_cell = np.zeros((len(batch_gen), batch_gen.label_seq_len, 2))
    running_loss = 0
    count = 0
    for grid, label_grid in batch_gen.batch_next():
        count += 1
        print('\rtest:{}'.format(count), flush=True, end='')
        label_grid = label_grid.permute(0, 1, 4, 2, 3).cpu().numpy()
        pred = model.predict(grid)

        selected_cell[count-1, :, 0] = label_grid[0, :, 0, 20, 20]
        selected_cell[count-1, :, 1] = pred[0, :, 0, 20, 20]
        _plot_heat_map(heat_map, label_grid, pred, count=count)

        running_loss += model.score(label_grid, pred)
    _plot_time_plot(time_plot, selected_cell)
    return running_loss / count


def _plot_heat_map(in_figure, label, pred, count):
    plt.figure(in_figure.number)
    plt.clf()
    for i in range(0, 10, 2):
        max_value = np.max(label[0, int(i/2), 0, :, :])
        min_value = np.min(label[0, int(i/2), 0, :, :])
        plt.subplot(5, 2, i+1)
        plt.imshow(label[0, int(i/2), 0, :, :],
                   cmap=plt.cm.get_cmap('Reds', 10),
                   interpolation='nearest',
                   vmin=min_value,
                   vmax=max_value)
        plt.xticks(())
        plt.yticks(())

        plt.subplot(5, 2, i+2)
        plt.imshow(pred[0, int(i/2), 0, :, :],
                   cmap=plt.cm.get_cmap('Reds', 10),
                   interpolation='nearest',
                   vmin=min_value,
                   vmax=max_value)
        plt.xticks(())
        plt.yticks(())
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.savefig('results/figures/heat_map_{}.png'.format(count),
                bbox_inches="tight", pad_inches=0, dpi=500)


def _plot_time_plot(time_plot, in_data):
    flatten_data = in_data.reshape(-1, 2)
    hours = np.arange(len(flatten_data))
    plt.figure(time_plot.number)
    plt.plot(hours, flatten_data[:, 0])
    plt.plot(hours, flatten_data[:, 1])
    plt.title('Seçilmiş Hücrenin Zaman Grafiği')
    plt.xlabel('Saat')
    plt.ylabel('Sıcaklık')
    plt.legend(['Gerçek Veri', 'Tahmin Edilmiş'])
    plt.grid(True)
    plt.savefig('results/figures/time_plot.png',
                bbox_inches="tight", pad_inches=0.1, dpi=500)

