import math
import numpy as np
from IPython.display import clear_output
from tqdm import tqdm_notebook as tqdm

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy.random as npr
import seaborn as sns
sns.color_palette("bright")
import matplotlib as mpl
import matplotlib.cm as cm
from Neural_ODE import *
import torch
from torch import Tensor
from torch import nn
from torch.nn  import functional as F
from torch.autograd import Variable

use_cuda = torch.cuda.is_available()


# This file focus on Latent ODE model which use
# RNN as decoder and ODE as encoder to extrapolate
# random generative spiral data
class LinearODEF(ODEF):
    def __init__(self, W):
        super(LinearODEF, self).__init__()
        self.lin = nn.Linear(2, 2, bias=False)
        self.lin.weight = nn.Parameter(W)

    def forward(self, x, t):
        return self.lin(x)
class TestODEF(ODEF):
    def __init__(self, A, B, x0):
        super(TestODEF, self).__init__()
        self.A = nn.Linear(2, 2, bias=False)
        self.A.weight = nn.Parameter(A)
        self.B = nn.Linear(2, 2, bias=False)
        self.B.weight = nn.Parameter(B)
        self.x0 = nn.Parameter(x0)

    def forward(self, x, t):
        xTx0 = torch.sum(x*self.x0, dim=1)
        dxdt = torch.sigmoid(xTx0) * self.A(x - self.x0) + torch.sigmoid(-xTx0) * self.B(x + self.x0)
        return dxdt
class NNODEF(ODEF):
    def __init__(self, in_dim, hid_dim, time_invariant=False):
        super(NNODEF, self).__init__()
        self.time_invariant = time_invariant

        if time_invariant:
            self.lin1 = nn.Linear(in_dim, hid_dim)
        else:
            self.lin1 = nn.Linear(in_dim+1, hid_dim)
        self.lin2 = nn.Linear(hid_dim, hid_dim)
        self.lin3 = nn.Linear(hid_dim, in_dim)
        self.elu = nn.ELU(inplace=True)

    def forward(self, x, t):
        if not self.time_invariant:
            x = torch.cat((x, t), dim=-1)

        h = self.elu(self.lin1(x))
        h = self.elu(self.lin2(h))
        out = self.lin3(h)
        return out

class RNNEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(RNNEncoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        self.rnn = nn.GRU(input_dim+1, hidden_dim)
        self.hid2lat = nn.Linear(hidden_dim, 2*latent_dim)

    def forward(self, x, t):
        # Concatenate time to input
        t = t.clone()
        t[1:] = t[:-1] - t[1:]
        t[0] = 0.
        xt = torch.cat((x, t), dim=-1)

        _, h0 = self.rnn(xt.flip((0,)))  # Reversed
        # Compute latent dimension
        z0 = self.hid2lat(h0[0])
        z0_mean = z0[:, :self.latent_dim]
        z0_log_var = z0[:, self.latent_dim:]
        return z0_mean, z0_log_var
class NeuralODEDecoder(nn.Module):
    def __init__(self, output_dim, hidden_dim, latent_dim):
        super(NeuralODEDecoder, self).__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        func = NNODEF(latent_dim, hidden_dim, time_invariant=True)
        self.ode = NeuralODE(func)
        self.l2h = nn.Linear(latent_dim, hidden_dim)
        self.h2o = nn.Linear(hidden_dim, output_dim)

    def forward(self, z0, t):
        zs = self.ode(z0, t, return_whole_sequence=True)

        hs = self.l2h(zs)
        xs = self.h2o(hs)
        return xs
class ODEVAE(nn.Module):
    def __init__(self, output_dim, hidden_dim, latent_dim):
        super(ODEVAE, self).__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        self.encoder = RNNEncoder(output_dim, hidden_dim, latent_dim)
        self.decoder = NeuralODEDecoder(output_dim, hidden_dim, latent_dim)

    def forward(self, x, t, MAP=False):
        z_mean, z_log_var = self.encoder(x, t)
        if MAP:
            z = z_mean
        else:
            z = z_mean + torch.randn_like(z_mean) * torch.exp(0.5 * z_log_var)
        x_p = self.decoder(z, t)
        return x_p, z, z_mean, z_log_var

    def generate_with_seed(self, seed_x, t):
        seed_t_len = seed_x.shape[0]
        z_mean, z_log_var = self.encoder(seed_x, t[:seed_t_len])
        x_p = self.decoder(z_mean, t)
        return x_p


def gen_batch(batch_size, n_sample=100):
    n_batches = samp_trajs.shape[1] // batch_size
    time_len = samp_trajs.shape[0]
    n_sample = min(n_sample, time_len)
    for i in range(n_batches):
        if n_sample > 0:
            t0_idx = npr.multinomial(1, [1. / (time_len - n_sample)] * (time_len - n_sample))
            t0_idx = np.argmax(t0_idx)
            tM_idx = t0_idx + n_sample
        else:
            t0_idx = 0
            tM_idx = time_len

        frm, to = batch_size * i, batch_size * (i + 1)
        yield samp_trajs[t0_idx:tM_idx, frm:to], samp_ts[t0_idx:tM_idx, frm:to]


def add_time(in_tensor, t):
    bs, c, w, h = in_tensor.shape
    return torch.cat((in_tensor, t.expand(bs, 1, w, h)), dim=1)


def to_np(x):
    return x.detach().cpu().numpy()


if __name__ == '__main__':
    t_max = 6.29*5
    n_points = 200
    noise_std = 0.02
    num_spirals = 1000
    n_epochs = 20000
    batch_size = 100
    plot_traj_idx = 1
    index_np = np.arange(0, n_points, 1, dtype=np.int)
    index_np = np.hstack([index_np[:, None]])
    times_np = np.linspace(0, t_max, num=n_points)
    times_np = np.hstack([times_np[:, None]] * num_spirals)
    times = torch.from_numpy(times_np[:, :, None]).to(torch.float32)

    # Generate random spirals parameters
    normal01 = torch.distributions.Normal(0, 1.0)

    x0 = Variable(normal01.sample((num_spirals, 2))) * 2.0

    W11 = -0.1 * normal01.sample((num_spirals,)).abs() - 0.05
    W22 = -0.1 * normal01.sample((num_spirals,)).abs() - 0.05
    W21 = -1.0 * normal01.sample((num_spirals,)).abs()
    W12 =  1.0 * normal01.sample((num_spirals,)).abs()

    xs_list = []
    for i in range(num_spirals):
        if i % 2 == 1: #  Make it counter-clockwise
            W21, W12 = W12, W21

        func = LinearODEF(Tensor([[W11[i], W12[i]], [W21[i], W22[i]]]))
        ode = NeuralODE(func)

        xs = ode(x0[i:i+1], times[:, i:i+1], return_whole_sequence=True)
        xs_list.append(xs)
    orig_trajs = torch.cat(xs_list, dim=1).detach()
    samp_trajs = orig_trajs + torch.randn_like(orig_trajs) * noise_std
    samp_ts = times
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 9))
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        ax.scatter(samp_trajs[:, i, 0], samp_trajs[:, i, 1], c=samp_ts[:, i, 0], cmap=cm.plasma)
    plt.show()
    plt.savefig('results/origin.png')
    vae = ODEVAE(2, 64, 6)
    #vae = vae.cuda()
    if use_cuda:
        vae = vae.cuda()
    optim = torch.optim.Adam(vae.parameters(), betas=(0.9, 0.999), lr=0.001)
    preload = False
    plot_traj = orig_trajs[:, plot_traj_idx:plot_traj_idx+1]
    plot_obs = samp_trajs[:, plot_traj_idx:plot_traj_idx+1]
    plot_ts = samp_ts[:, plot_traj_idx:plot_traj_idx+1]
    if use_cuda:
        plot_traj = plot_traj.cuda()
        plot_obs = plot_obs.cuda()
        plot_ts = plot_ts.cuda()

    if preload:
        vae.load_state_dict(torch.load("models/vae_spirals.sd"))

    for epoch_idx in range(n_epochs):
        losses = []
        train_iter = gen_batch(batch_size)
        for x, t in train_iter:
            optim.zero_grad()
            if use_cuda:
                x, t = x.cuda(), t.cuda()

            max_len = np.random.choice([30, 50, 100])
            permutation = np.random.permutation(t.shape[0])
            np.random.shuffle(permutation)
            permutation = np.sort(permutation[:max_len])

            x, t = x[permutation], t[permutation]

            x_p, z, z_mean, z_log_var = vae(x, t)
            kl_loss = -0.5 * torch.sum(1 + z_log_var - z_mean**2 - torch.exp(z_log_var), -1)
            loss = 0.5 * ((x-x_p)**2).sum(-1).sum(0) / noise_std**2 + kl_loss
            loss = torch.mean(loss)
            loss /= max_len
            loss.backward()
            optim.step()
            losses.append(loss.item())

        print(f"Epoch {epoch_idx}")

        frm, to, to_seed = 0, 200, 50
        seed_trajs = samp_trajs[frm:to_seed]
        ts = samp_ts[frm:to]
        if use_cuda:
            seed_trajs = seed_trajs.cuda()
            ts = ts.cuda()

        samp_trajs_p = to_np(vae.generate_with_seed(seed_trajs, ts))
        print(np.mean(losses), np.median(losses))
        clear_output(wait=True)
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 9))
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        ax.scatter(to_np(seed_trajs[:, i, 0]), to_np(seed_trajs[:, i, 1]), c=to_np(ts[frm:to_seed, i, 0]), cmap=cm.plasma)
        ax.plot(to_np(orig_trajs[frm:to, i, 0]), to_np(orig_trajs[frm:to, i, 1]))
        ax.plot(samp_trajs_p[:, i, 0], samp_trajs_p[:, i, 1])
    plt.show()
    plt.savefig('results/predict.png')
    spiral_0_idx = 3
    spiral_1_idx = 6

    homotopy_p = Tensor(np.linspace(0., 1., 10)[:, None])
    vae = vae
    if use_cuda:
        homotopy_p = homotopy_p.cuda()
        vae = vae.cuda()

    spiral_0 = orig_trajs[:, spiral_0_idx:spiral_0_idx+1, :]
    spiral_1 = orig_trajs[:, spiral_1_idx:spiral_1_idx+1, :]
    ts_0 = samp_ts[:, spiral_0_idx:spiral_0_idx+1, :]
    ts_1 = samp_ts[:, spiral_1_idx:spiral_1_idx+1, :]
    if use_cuda:
        spiral_0, ts_0 = spiral_0.cuda(), ts_0.cuda()
        spiral_1, ts_1 = spiral_1.cuda(), ts_1.cuda()

    z_cw, _ = vae.encoder(spiral_0, ts_0)
    z_cc, _ = vae.encoder(spiral_1, ts_1)

    homotopy_z = z_cw * (1 - homotopy_p) + z_cc * homotopy_p

    t = torch.from_numpy(np.linspace(0, 6*np.pi, 200))
    t = t[:, None].expand(200, 10)[:, :, None]
    t = t.cuda() if use_cuda else t
    hom_gen_trajs = vae.decoder(homotopy_z, t)

    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(15, 5))
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        ax.plot(to_np(hom_gen_trajs[:, i, 0]), to_np(hom_gen_trajs[:, i, 1]))
    plt.show()
    plt.savefig('results/homo.png')