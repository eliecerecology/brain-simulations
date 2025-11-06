import torch
import numpy as np
from typing import List
import tqdm

Complex = torch.complex128

class KuramotoFast:
    def __init__(self, n_nodes: int, n_oscillators: int, sampling_rate: int, k_list: List[float], weight_matrix: np.ndarray,
                 frequency_spread: float, noise_scale: float=1.0, use_cuda: bool=True, use_tqdm: bool=True, node_frequencies=None, **kwargs):
        """
            Implementation of nested Kuramoto model using PyTorch. The model consists of N nodes each with M oscillators. Each pair of nodes is connected with directed weight given by weight_matrix.

            :param n_nodes: number of nodes in the model
            :param n_oscillators: number of oscillators in each node
            :param sampling_rate: update rate of the model
            :param k_list: list of K values (within node shift) of the model. Should have length equal to number of nodes.
            :param weight_matrix: 2d matrix of node vs node connectivity weight. Should have N_nodes x N_nodes shape.
            :param frequency_spread: spread of frequencies within a node. Frequencies of oscillators are defined as linspace from central_frequency - frequency_spread to central_frequency + frequency_spread
            :param noise_scale: sigma of noise.
            :param use_cuda: use GPU (PyTorch CUDA/ROCm) to compute the model?
        """
        self._check_parameters(n_nodes, k_list, weight_matrix)

        # Set device
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

        # Convert weight_matrix to PyTorch tensor
        self.weight_matrix = torch.tensor(weight_matrix, dtype=torch.float64, device=self.device)
        torch.diagonal(self.weight_matrix).fill_(0)  # Set diagonal to 0
        self.weight_matrix = self.weight_matrix.T  # Transpose for directed connectivity

        self.sampling_rate = sampling_rate
        self.dt = 1.0 / sampling_rate
        self.use_cuda = use_cuda
        self.disable_tqdm = not(use_tqdm)

        self._init_parameters()
        self._preallocate()

    def _check_parameters(self, n_nodes: int, k_list: List[float], weight_matrix: np.ndarray):
        if len(k_list) != n_nodes:
            raise RuntimeError(f'Size of k_list ({len(k_list)}) is not equal to number of nodes ({n_nodes}).')
        if np.ndim(weight_matrix) != 2 or (weight_matrix.shape[0] != weight_matrix.shape[1]):
            raise RuntimeError(f'weight_matrix should be a 2d square matrix, got {weight_matrix.shape} shape.')
        if weight_matrix.shape[0] != n_nodes or weight_matrix.shape[1] != n_nodes:
            raise RuntimeError(f'weight matrix should be a 2d matrix of size N_nodes x N_nodes, got {weight_matrix.shape} shape')

    def _init_parameters(self):
        # Angular frequencies in rad/s
        omegas = torch.zeros((self.n_nodes, self.n_oscillators), dtype=torch.float64, device=self.device)
        for idx, frequency in enumerate(self.node_frequencies):
            freq_lower = frequency - self.frequency_spread
            freq_upper = frequency + self.frequency_spread
            omegas[idx] = torch.linspace(freq_lower, freq_upper, steps=self.n_oscillators, device=self.device, dtype=torch.float64)

        omegas += torch.rand(omegas.shape, device=self.device, dtype=torch.float64) * 0.2 - 0.1  # Uniform [-0.1, 0.1]
        self.omegas = omegas * 2 * np.pi  # rad/s

        # Shift coeffs
        C = torch.tensor(self.k_list, dtype=torch.float64, device=self.device) / self.n_oscillators
        self.shift_coeffs = C.view(-1, 1)

        # Random initial phase
        thetas = torch.rand(omegas.shape, device=self.device, dtype=torch.float64) * 2 * np.pi - np.pi  # Uniform [-π, π]
        self.phases = torch.exp(1j * thetas)

        self._complex_dtype = torch.complex64
        self._float_dtype = torch.float32

    def _preallocate(self):
        n_nodes, n_osc = self.phases.shape
        self._phase_conj = torch.empty_like(self.phases)
        self._external_buffer = torch.empty((n_nodes, n_nodes, n_osc), dtype=self.phases.dtype, device=self.device)

    def _compute_rhs(self, phases):
        mean_phase = torch.mean(phases, dim=1)  # Shape: (n_nodes,)
        self._phase_conj = torch.conj(phases)  # Shape: (n_nodes, n_oscillators)

        # External dynamics
        # tensordot computes outer product: (n_nodes, n_oscillators) × (n_nodes,) → (n_nodes, n_oscillators, n_nodes)
        # permute reorders to (n_nodes, n_nodes, n_oscillators)
        self._external_buffer = torch.tensordot(self._phase_conj, mean_phase, dims=0).permute(0, 2, 1)
        weight_expanded = self.weight_matrix[:, :, None].expand(-1, -1, self.n_oscillators)  # Shape: (n_nodes, n_nodes, n_oscillators)
        self._external_buffer *= weight_expanded
        external = self._external_buffer.sum(dim=1)  # Shape: (n_nodes, n_oscillators)
        external_rhs = external.imag / self.n_nodes

        # Internal dynamics
        self._phase_conj = phases * torch.sum(self._phase_conj, dim=1, keepdim=True)
        self._phase_conj = torch.conj(self._phase_conj)
        internal_rhs = self._phase_conj.imag * self.shift_coeffs

        rhs = self.omegas + internal_rhs + external_rhs
        return rhs

    def simulate(self, time: float, noise_realisations: int=100, random_seed: int=42) -> np.ndarray:
        """
            Implementation of nested Kuramoto model with RK4 integration using PyTorch.

            :param time: Length of the simulation in seconds. Total number of samples is computed as sampling_rate x time + 1 (initial state)
            :param noise_realisations: Number of noise realisations to generate (not used in this version)
            :return: N_nodes x N_ts matrix of complex values that contains each node activity during the simulation
        """
        torch.manual_seed(random_seed)
        n_iters = int(time * self.sampling_rate)
        history = torch.zeros((self.n_nodes, n_iters + 1), dtype=self._complex_dtype, device=self.device)
        history[:, 0] = self.phases.mean(dim=1)

        for i in tqdm.trange(1, n_iters + 1, leave=False, desc='Kuramoto model is running...', disable=self.disable_tqdm):
            # RK4 steps
            k1 = self._compute_rhs(self.phases)
            phases2 = self.phases * torch.exp(1j * (self.dt / 2) * k1)
            k2 = self._compute_rhs(phases2)
            phases3 = self.phases * torch.exp(1j * (self.dt / 2) * k2)
            k3 = self._compute_rhs(phases3)
            phases4 = self.phases * torch.exp(1j * self.dt * k3)
            k4 = self._compute_rhs(phases4)
            rhs = (k1 + 2 * k2 + 2 * k3 + k4) / 6

            shift_noise = torch.normal(mean=0.0, std=self.noise_scale, size=rhs.shape, device=self.device, dtype=torch.float64)
            rhs += shift_noise
            self.phases *= torch.exp(1j * self.dt * rhs)
            history[:, i] = self.phases.mean(dim=1)

        # Convert to NumPy for return
        history = history.cpu().numpy()
        return history