import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class TrajGRUCell(nn.Module):
    def __init__(self, input_size, input_dim, hidden_dim,
                 kernel_size, bias, connection):
        """
        :param input_size: (int, int) width(M) and height(N) of input grid
        :param input_dim: int, number of channels (D) of input grid
        :param hidden_dim: int, number of channels of hidden state
        :param kernel_size: (int, int) size of the convolution kernel
        :param bias: bool weather or not to add the bias
        """
        super(TrajGRUCell, self).__init__()

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
        """
        :param x: (b, d, m, n)
        :param h_prev: (b, d, m, n)
        :return: (b, d, m, n)
        """
        input_conv = self.conv_input(x)

        x_z, x_r, x_h = torch.split(input_conv, self.hidden_dim, dim=1)

        for local_link, warped in enumerate(self.__warp(x, h_prev)):
            if local_link == 0:
                traj_tensor = self.projecting_channels[local_link](warped)
            else:
                traj_tensor += self.projecting_channels[local_link](warped)

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
        # initialized to zero, for hidden state of GRU
        :param batch_size: int
        :return:(b, d, m, n) tensor
        """
        hidden = Variable(torch.zeros(batch_size, self.hidden_dim, self.height, self.width))
        hidden = hidden.to(device)
        return hidden


class EncoderBlock(nn.Module):
    def __init__(self, **kwargs):
        super(EncoderBlock, self).__init__()
        # encoder conf
        self.input_size = kwargs['input_size']
        self.input_dim = kwargs['input_dim']
        self.num_layers = kwargs['num_layers']

        # down-sample conf
        self.conv_dims = kwargs['conv_dims']
        self.conv_kernel = kwargs['conv_kernel']
        self.conv_stride = kwargs['conv_stride']
        self.conv_padding = self.conv_kernel // 2

        # traj-gru conf
        self.gru_input_sizes = self.__calc_input_size()
        self.gru_dims = kwargs['gru_dims']
        self.gru_kernels = kwargs['gru_kernels']
        self.connection = kwargs['connection']
        self.bias = kwargs['bias']

        self.cell_list = []
        for i in range(self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.conv_dims[i-1]
            self.cell_list += [
                nn.Conv2d(in_channels=cur_input_dim,
                          out_channels=self.conv_dims[i],
                          kernel_size=self.conv_kernel,
                          stride=self.conv_stride,
                          padding=self.conv_padding),

                TrajGRUCell(input_size=self.gru_input_sizes[i],
                            input_dim=self.conv_dims[i],
                            hidden_dim=self.gru_dims[i],
                            kernel_size=self.gru_kernels[i],
                            connection=self.connection,
                            bias=self.bias)
            ]
        self.cell_list = nn.ModuleList(self.cell_list)

    def init_memory(self, batch_size):
        """
        Initialise every memory element hidden state
        :param batch_size: int
        :return: list of tensors (b, d, m, n)
        """
        init_states = []
        # Only iterate odd indexes
        for i in range(1, 2*self.num_layers+1, 2):
            init_states.append(self.cell_list[i].init_hidden(batch_size))
        return init_states

    def forward(self, input_tensor, hidden_states):
        """
        :param input_tensor: (B, D, M, N)
        :param hidden_states: (B, D, M, N)
        :return:(B, D', M', N') down-sampled tensor
        """
        layer_state_list = []

        cur_layer_input = input_tensor
        for layer_idx in range(self.num_layers):
            # Down-sample
            conv_output = self.cell_list[layer_idx](cur_layer_input)

            # Memory element
            cur_layer_input = self.cell_list[layer_idx+1](conv_output,
                                                          hidden_states[layer_idx])
            layer_state_list.append(cur_layer_input)

        return layer_state_list

    def __calc_input_size(self):
        input_sizes = []
        cur_dim = self.input_size
        for i in range(self.num_layers):
            h, w = cur_dim
            f = self.conv_kernel
            s = self.conv_stride
            p = self.conv_padding
            cur_dim = ((h - f + 2*p)/s + 1, (w - f + 2*p)/s + 1)
            input_sizes.append(cur_dim)
        return input_sizes


class DecoderBlock(nn.Module):
    def __init__(self, **kwargs):
        super(DecoderBlock, self).__init__()

        # decoder conf
        self.input_size = kwargs['input_size']
        self.input_dim = kwargs['input_dim']
        self.num_layers = kwargs['num_layers']

        # up-sample conf
        self.conv_dims = kwargs['conv_dims']
        self.conv_kernel = kwargs['conv_kernel']
        self.conv_stride = kwargs['conv_stride']
        self.conv_padding = self.conv_kernel // 2

        # traj-gru conf
        self.gru_input_sizes = self.__calc_input_size()
        self.gru_dims = kwargs['gru_dims']
        self.gru_kernels = kwargs['gru_kernels']
        self.connection = kwargs['connection']
        self.bias = kwargs['bias']

        # output convs conf
        self.output_dim = kwargs['output_dim']

        self.cell_list = []
        for i in range(self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.conv_dims[i-1]
            cur_input_size = self.input_size if i == 0 else self.gru_input_sizes[i-1]
            self.cell_list += [
                TrajGRUCell(input_size=cur_input_size,
                            input_dim=cur_input_dim,
                            hidden_dim=self.gru_dims[i],
                            kernel_size=self.gru_kernels[i],
                            connection=self.connection,
                            bias=self.bias),

                nn.ConvTranspose2d(in_channels=self.gru_dims[i],
                                   out_channels=self.conv_dims[i],
                                   kernel_size=self.conv_kernel,
                                   stride=self.conv_stride,
                                   padding=self.conv_padding),
            ]
        self.cell_list = nn.ModuleList(self.cell_list)

        self.output_convs = nn.Sequential(
            nn.Conv2d(in_channels=self.conv_dims[-1],
                      out_channels=self.conv_dims[-1],
                      kernel_size=self.conv_kernel,
                      stride=1,
                      padding=self.conv_kernel // 2),
            nn.Conv2d(in_channels=self.conv_dims[-1],
                      out_channels=self.output_dim,
                      kernel_size=1,
                      stride=1)
        )

    def init_memory(self, batch_size):
        """
        Initialise every memory element hidden state
        :param batch_size: int
        :return: list of tensors (b, d, m, n)
        """
        init_states = []
        # Only iterate even indexes
        for i in range(0, 2*self.num_layers, 2):
            init_states.append(self.cell_list[i].init_hidden(batch_size))
        return init_states

    def forward(self, input_tensor, hidden_states):
        """
        :param input_tensor: (B, D, M, N)
        :param hidden_states: (B, D, M, N)
        :return:(B, D', M', N') down-sampled tensor
        """
        layer_state_list = []

        cur_layer_input = input_tensor
        for layer_idx in range(self.num_layers):
            # Down-sample
            conv_output = self.cell_list[layer_idx](cur_layer_input)

            # Memory element
            cur_layer_input = self.cell_list[layer_idx+1](conv_output,
                                                          hidden_states[layer_idx])
            layer_state_list.append(cur_layer_input)

        output = self.output_convs(cur_layer_input)

        return output, layer_state_list

    def __calc_input_size(self):
        input_sizes = []
        cur_dim = self.input_size
        for i in range(self.num_layers):
            h, w = cur_dim
            f = self.conv_kernel
            s = self.conv_stride
            p = self.conv_padding
            cur_dim = ((h-1)*s - 2*p + f, (w-1)*s - 2*p + f)
            input_sizes.append(cur_dim)
        return input_sizes


class TrajGRU(nn.Module):
    def __init__(self, **kwargs):
        super(TrajGRU, self).__init__()

        self.encoder_count = kwargs['encoder_count']
        self.decoder_count = kwargs['decoder_count']

        self.encoder_conf = kwargs['encoder_conf']
        self.decoder_conf = kwargs['decoder_conf']

        self.encoder = []
        for i in range(self.encoder_count):
            self.encoder.append(
                EncoderBlock()
            )


if __name__ == '__main__':

    traj_gru = TrajGRUCell(input_size=(21, 41),
                           input_dim=5,
                           hidden_dim=5,
                           kernel_size=5,
                           connection=17,
                           bias=True)

    encoder_block = EncoderBlock(input_size=(21, 41),
                                 input_dim=5,
                                 num_layers=2,
                                 conv_dims=[16, 64],
                                 conv_kernel=3,
                                 conv_stride=2,
                                 gru_dims=[32, 96],
                                 gru_kernels=[5, 3],
                                 connection=5,
                                 bias=True)

    decoder_block = DecoderBlock(input_size=(6, 11),
                                 input_dim=96,
                                 output_dim=3,
                                 num_layers=2,
                                 conv_dims=[64, 16],
                                 conv_kernel=3,
                                 conv_stride=2,
                                 gru_dims=[96, 32],
                                 gru_kernels=[3, 3],
                                 connection=5,
                                 bias=True)

    # print(traj_gru)
    # print(encoder_block)
    print(decoder_block)

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


