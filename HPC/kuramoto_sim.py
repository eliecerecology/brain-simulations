import numpy as np
import time
import torch
import sys
import scipy
import matplotlib.pyplot as plt
from scipy import signal
import networkx as nx
from scipy.stats import nbinom
import secrets
from numpy.polynomial.polynomial import polyfit, polyval
from typing import List
import tqdm
import os

class KuramotoFast:
    def __init__(self, n_nodes: int, n_oscillators: int, sampling_rate: int, k_list: List[float], weight_matrix: np.ndarray,
                 frequency_spread: float, noise_scale: float=1.0, use_cuda: bool=True, use_tqdm: bool=True, node_frequencies=None, **kwargs):
        self._check_parameters(n_nodes, k_list, weight_matrix)

        if use_cuda and torch.cuda.is_available():
            self.device = torch.device("cuda")
            device_name = "GPU (CUDA/ROCm)"
        else:
            self.device = torch.device("cpu")
            device_name = "CPU"
        print(f"Using device: {device_name}")

        self.n_nodes = n_nodes
        self.n_oscillators = n_oscillators
        self.k_list = k_list
        self.noise_scale = 2 * np.pi * noise_scale / sampling_rate
        self.frequency_spread = frequency_spread
        self.node_frequencies = node_frequencies

        self.weight_matrix = torch.tensor(weight_matrix, dtype=torch.float32, device=self.device)
        torch.diagonal(self.weight_matrix).fill_(0)
        self.weight_matrix = self.weight_matrix.T

        self.sampling_rate = sampling_rate
        self.dt = 1.0 / sampling_rate
        self.use_cuda = use_cuda
        self.disable_tqdm = not(use_tqdm)

        self._init_parameters()
        self._preallocate()

    def _check_parameters(self, n_nodes, k_list, weight_matrix):
        if len(k_list) != n_nodes:
            raise RuntimeError(f'Size of k_list ({len(k_list)}) is not equal to number of nodes ({n_nodes}).')
        if np.ndim(weight_matrix) != 2 or (weight_matrix.shape[0] != weight_matrix.shape[1]):
            raise RuntimeError(f'weight_matrix should be a 2d square matrix, got {weight_matrix.shape} shape.')
        if weight_matrix.shape[0] != n_nodes:
            raise RuntimeError(f'weight matrix should be N_nodes x N_nodes, got {weight_matrix.shape}')

    def _init_parameters(self):
        omegas = torch.zeros((self.n_nodes, self.n_oscillators), dtype=torch.float32, device=self.device)
        for idx, frequency in enumerate(self.node_frequencies):
            freq_lower = frequency - self.frequency_spread
            freq_upper = frequency + self.frequency_spread
            omegas[idx] = torch.linspace(freq_lower, freq_upper, steps=self.n_oscillators, device=self.device, dtype=torch.float32)

        omegas += torch.rand(omegas.shape, device=self.device, dtype=torch.float32) * 0.2 - 0.1
        self.omegas = omegas * 2 * np.pi

        C = torch.tensor(self.k_list, dtype=torch.float32, device=self.device) / self.n_oscillators
        self.shift_coeffs = C.view(-1, 1)

        thetas = torch.rand(omegas.shape, device=self.device, dtype=torch.float32) * 2 * np.pi - np.pi
        self.phases = torch.exp(1j * thetas).to(torch.complex64)

        self._complex_dtype = torch.complex64
        self._float_dtype = torch.float32

    def _preallocate(self):
        n_nodes, n_osc = self.phases.shape
        self._phase_conj = torch.empty_like(self.phases)
        self._external_buffer = torch.empty((n_nodes, n_nodes, n_osc), dtype=self.phases.dtype, device=self.device)

    def _compute_rhs(self, phases):
        mean_phase = torch.mean(phases, dim=1)
        self._phase_conj = torch.conj(phases)

        self._external_buffer = torch.tensordot(self._phase_conj, mean_phase, dims=0).permute(0, 2, 1)
        weight_expanded = self.weight_matrix[:, :, None].expand(-1, -1, self.n_oscillators)
        self._external_buffer *= weight_expanded
        external = self._external_buffer.sum(dim=1)
        external_rhs = external.imag / self.n_nodes

        self._phase_conj = phases * torch.sum(self._phase_conj, dim=1, keepdim=True)
        self._phase_conj = torch.conj(self._phase_conj)
        internal_rhs = self._phase_conj.imag * self.shift_coeffs

        rhs = self.omegas + internal_rhs + external_rhs
        return rhs

    def _rotate(self, dtheta):
        return torch.polar(torch.ones_like(dtheta), dtheta)

    def simulate(self, time: float, noise_realisations: int=100, random_seed: int=42) -> np.ndarray:
        torch.manual_seed(random_seed)
        n_iters = int(time * self.sampling_rate)
        history = torch.zeros((self.n_nodes, n_iters + 1), dtype=self._complex_dtype, device=self.device)
        history[:, 0] = self.phases.mean(dim=1)
        for i in tqdm.trange(1, n_iters + 1, leave=False, desc='Kuramoto model is running...', disable=self.disable_tqdm):
            k1 = self._compute_rhs(self.phases)
            phases2 = self.phases * self._rotate((self.dt / 2) * k1)
            k2 = self._compute_rhs(phases2)
            phases3 = self.phases * self._rotate((self.dt / 2) * k2)
            k3 = self._compute_rhs(phases3)
            phases4 = self.phases * self._rotate(self.dt * k3)
            k4 = self._compute_rhs(phases4)
            rhs = (k1 + 2 * k2 + 2 * k3 + k4) / 6
            shift_noise = torch.normal(
                mean=0.0, std=self.noise_scale, size=rhs.shape,
                device=self.device, dtype=torch.float32,
            )
            rhs += shift_noise
            self.phases = self.phases * self._rotate(self.dt * rhs)
            history[:, i] = self.phases.mean(dim=1)
        history = history.cpu().numpy()
        return history


def FA_metric(phasor, scales):
    y = calc_detrened(phasor)
    F_fa = np.zeros(len(scales))
    for i, s in enumerate(scales):
        diffs = y[s:] - y[:-s]
        F_fa[i] = np.sqrt(np.mean(diffs**2))
    coeff_fa = np.polyfit(np.log2(scales), np.log2(F_fa), 1)
    alpha_fa = coeff_fa[0]
    fit_fa = 2 ** np.polyval(coeff_fa, np.log2(scales))
    print(f"Estimated FA exponent α = {alpha_fa:.3f}")
    return fit_fa, alpha_fa

def calc_detrened(data):
    x = np.abs(data)
    y = np.cumsum(x - np.mean(x))
    return y

def dfa_rms(y, scale):
    n_windows = len(y) // scale
    if n_windows == 0:
        return np.nan
    shape = (n_windows, scale)
    Y = np.lib.stride_tricks.as_strided(y, shape=shape)
    rms = np.zeros(n_windows)
    scale_axis = np.arange(scale)
    for i, window in enumerate(Y):
        coeff = np.polyfit(scale_axis, window, 1)
        trend = np.polyval(coeff, scale_axis)
        rms[i] = np.sqrt(np.mean((window - trend) ** 2))
    return np.mean(rms)

def dfa_scales(min_exp=5, max_exp=9, step=0.25):
    scales = np.round(2 ** np.arange(min_exp, max_exp, step)).astype(int)
    scales = np.unique(scales)
    return scales

def DFA(data):
    y = calc_detrened(data)
    scales = dfa_scales()
    F = []
    for s in scales:
        rms_val = dfa_rms(y, s)
        F.append(rms_val if not np.isnan(rms_val) else np.nan)
    F = np.array(F)
    mask = ~np.isnan(F)
    scales = scales[mask]
    F = F[mask]
    coeff = np.polyfit(np.log2(scales), np.log2(F), 1)
    alpha = coeff[0]
    return alpha, scales, F

def plv_matrix_vectorized(inst_theta):
    X = np.exp(1j * inst_theta)
    M = np.dot(X.conj().T, X) / X.shape[0]
    return np.abs(M)

def run_simulation_batch(i_start, i_end, all_adj, k_values, n_nodes, n_oscillators,
                          sampling_rate, frequency_spread, node_frequencies,
                          use_cuda, sim_time, network_names, output_dir, num_replicas):
    n_batch = i_end - i_start
    n_k = len(k_values)

    order_matrix     = np.zeros((n_batch, num_replicas, n_k))
    variability_matrix = np.zeros((n_batch, num_replicas, n_k))
    plv_matrix       = np.zeros((n_batch, num_replicas, n_k))
    dfa_matrix       = np.zeros((n_batch, num_replicas, n_k))
    fa_matrix        = np.zeros((n_batch, num_replicas, n_k))

    # Remove first 10 seconds as transient = 2000 samples = 100 oscillations
    transient_samples = int(10 * sampling_rate)
    print(f"Removing {transient_samples} samples as transient ({transient_samples/sampling_rate:.0f} seconds, "
          f"{transient_samples/sampling_rate * node_frequencies[0]:.0f} oscillations)")

    for local_i, i in enumerate(range(i_start, i_end)):
        print(f"\nNetwork {i}: {network_names[i]}")
        for r in range(num_replicas):
            print(f"  replica {r + 1}/{num_replicas}")
            W = all_adj[i, r]

            for k_idx, k in enumerate(k_values):
                random_seed = int(np.random.randint(0, 1_000_000))

                model = KuramotoFast(
                    n_nodes=n_nodes,
                    n_oscillators=n_oscillators,
                    k_list=[k] * n_nodes,
                    weight_matrix=W,
                    node_frequencies=node_frequencies,
                    sampling_rate=sampling_rate,
                    frequency_spread=frequency_spread,
                    use_cuda=use_cuda,
                )
                phase_data = model.simulate(time=sim_time, random_seed=random_seed)
                del model
                if use_cuda:
                    torch.cuda.empty_cache()

                # Remove transient — first 10 seconds = 2000 samples = 100 oscillations
                phase_data = phase_data[:, transient_samples:]

                order_ts   = np.abs(np.mean(phase_data, axis=0))
                order_mean = order_ts.mean()
                order_std  = order_ts.std()

                inst_theta = np.angle(phase_data).T
                plv_mat    = plv_matrix_vectorized(inst_theta)
                np.fill_diagonal(plv_mat, 0)
                plv_order  = plv_mat[np.triu_indices_from(plv_mat, k=1)].mean()

                scales       = dfa_scales()
                global_phasor = np.abs(phase_data).mean(axis=0)
                alpha        = DFA(global_phasor)
                fa           = FA_metric(global_phasor, scales)

                order_matrix[local_i, r, k_idx]      = order_mean
                variability_matrix[local_i, r, k_idx] = order_std
                plv_matrix[local_i, r, k_idx]         = plv_order
                dfa_matrix[local_i, r, k_idx]         = alpha[0]
                fa_matrix[local_i, r, k_idx]          = fa[1]

    results = {
        "order_matrix":      order_matrix,
        "variability_matrix": variability_matrix,
        "plv_matrix":        plv_matrix,
        "dfa_matrix":        dfa_matrix,
        "fa_matrix":         fa_matrix,
        "k_values":          k_values,
        "i_start":           i_start,
        "i_end":             i_end,
        "network_names":     np.array(network_names[i_start:i_end], dtype=object),
    }

    # Save to output directory
    output_path = os.path.join(output_dir, f"results_networks_{i_start}_to_{i_end}.npz")
    np.savez(output_path, **results)
    print(f"\nResults saved to {output_path}")
    return results


if __name__ == "__main__":
    print("CUDA:", torch.cuda.is_available())

    # --- Parameters ---
    k_values          = np.linspace(1, 60, 15)
    n_nodes           = 1000
    n_oscillators     = 100
    sampling_rate     = 200
    frequency_spread  = 3
    node_frequencies  = [10.0] * n_nodes
    use_cuda          = torch.cuda.is_available()
    sim_time          = 30       # 30 seconds total
    # transient = first 10 sec = 2000 samples = 100 oscillations (removed inside function)
    # analysis  = last 20 sec  = 4000 samples = 200 oscillations

    # Output directory — saves results in thesis folder
    output_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"Results will be saved to: {output_dir}")

    network_names = [
        "LL", "LH", "HL", "HH",
        "Watts-Strogatz Low mean", "Watts-Strogatz High mean",
        "Erdos-Renyi Low mean", "Erdos-Renyi High mean",
        "Fully Connected",
        "Barabasi-Albert Low mean", "Barabasi-Albert High mean",
    ]

    # Load adjacency matrices
    data    = np.load(os.path.join(output_dir, "adjacency_matrices.npz"))
    all_adj = data["adjacency_matrices"]
    num_networks, num_replicas, N, _ = all_adj.shape
    assert num_networks == 11 and num_replicas == 3

    # --- Run simulation ---
    # Change i_start and i_end to run different networks
    # 0:4  = LL, LH, HL, HH
    # 4:9  = WS low, WS high, ER low, ER high, Fully Connected
    # 9:11 = BA low, BA high
    run_simulation_batch(
        i_start=0, i_end=11,          # run ALL networks
        all_adj=all_adj,
        k_values=k_values,
        n_nodes=n_nodes,
        n_oscillators=n_oscillators,
        sampling_rate=sampling_rate,
        frequency_spread=frequency_spread,
        node_frequencies=node_frequencies,
        use_cuda=use_cuda,
        sim_time=sim_time,
        network_names=network_names,
        output_dir=output_dir,
        num_replicas=num_replicas,
    )
    print("All simulations complete.")
