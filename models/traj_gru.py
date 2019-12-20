import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ConvGRU(nn.Module):

    def __init__(self, input_size, input_dim, hidden_dim,
                 kernel_size, bias, connection):
        """
        :param input_size: (int, int) width(M) and height(N) of input grid
        :param input_dim: int, number of channels (D) of input grid
        :param hidden_dim: int, number of channels of hidden state
        :param kernel_size: (int, int) size of the convolution kernel
        :param bias: bool weather or not to add the bias
        """
        super(ConvGRU, self).__init__()

        self.height, self.width = input_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.connection = connection

        self.kernel_size = kernel_size
        self.bias = bias
        self.padding = self.kernel_size // 2

        self.conv_input = nn.Conv2d(in_channels=self.input_dim,
                                    out_channels=3*self.hidden_dim,
                                    kernel_size=self.kernel_size,
                                    padding=self.padding,
                                    bias=self.bias)

        self.projecting_channels = []
        for l in range(self.connection):
            self.projecting_channels.append(nn.Conv2d(in_channels=self.hidden_dim,
                                                      out_channels=3*self.hidden_dim,
                                                      kernel_size=1))

        self.projecting_channels = nn.ModuleList(self.projecting_channels)

        self.sub_net = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                                 out_channels=2 * self.connection,
                                 kernel_size=5,
                                 padding=1)

    def forward(self, x, h_prev):

        input_conv = self.conv_input(x)

        x_z, x_r, x_h = torch.split(input_conv, self.hidden_dim, dim=1)

        for l, warped in enumerate(self.__warp(x, h_prev)):
            if l == 0:
                traj_tensor = self.projecting_channels[l](warped)
            else:
                traj_tensor += self.projecting_channels[l](warped)

        h_z, h_r, h_h = torch.split(traj_tensor, self.hidden_dim, dim=1)

        z = torch.sigmoid(x_z + h_z)
        r = torch.sigmoid(x_r + h_r)
        h = F.leaky_relu(x_h + r * h_h, negative_slope=0.2)

        h_next = (1 - z) * h + z * h_prev

        return h_next

    def __warp(self, x, h):
        """
        :param x: (b, d, m, n)
        :param h: (b, d, m, n)
        :return:
        """
        combined = torch.cat([x, h], dim=1)
        combined_conv = self.sub_net(combined)

        # (b, 2L, m, n) --> (b, m, n, 2L)
        combined_conv = combined_conv.permute(0, 2, 3, 1)

        # scale to [0, 1]
        combined_conv = (combined_conv - combined_conv.min()) /\
                        (combined_conv.max() - combined_conv.min())
        # scale to [-1, 1]
        combined_conv = 2 * combined_conv - 1

        for l in range(0, self.connection, 2):
            # (b, m, n, 2)
            grid = combined_conv[:, :, :, l:l+2]
            warped = F.grid_sample(h, grid, mode='bilinear')
            yield warped

    def init_hidden(self, batch_size):
        """
        # Create new tensor with sizes n_layers x batch_size x n_hidden,
        # initialized to zero, for hidden state of LSTM
        :param batch_size: int
        :return:
        """
        hidden = Variable(torch.zeros(batch_size, self.hidden_dim, self.height, self.width)).to(device)
        return hidden


if __name__ == '__main__':

    conv_gru = ConvGRU(input_size=(20, 40),
                       input_dim=5,
                       hidden_dim=5,
                       kernel_size=5,
                       connection=17,
                       bias=True)

    # from torchvision import transforms
    # from PIL import Image
    # from PIL import ImageFile
    #
    # ImageFile.LOAD_TRUNCATED_IMAGES = True
    #
    # im_path = '11.jpg'
    # image = Image.open(im_path)
    # image = image.convert('RGB')
    #
    # im_transform = transforms.Compose([
    #     transforms.CenterCrop(224),
    #     transforms.ToTensor()
    # ])
    # image = im_transform(image)
    # # Add batch dimension
    # image = image.unsqueeze(dim=0)
    #
    # d = torch.linspace(-1, 1, 224)
    # meshx, meshy = torch.meshgrid((d, d))
    #
    # # Just to see the effect
    # meshx = meshx * 0.3
    # meshy = meshy * 0.9
    #
    # grid = torch.stack((meshy, meshx), 2)
    # grid = grid.unsqueeze(0)
    # warped = F.grid_sample(image, grid, mode='bilinear', align_corners=False)
    #
    # to_image = transforms.ToPILImage()
    # to_image(image.squeeze()).show()
    # to_image(warped.squeeze()).show(title='WARPED')


