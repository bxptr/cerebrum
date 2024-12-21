import torch
import torch.nn as nn
from models.hh_solver import ExtendedHodgkinHuxleyNode
from models.integrator import rk4_step_extended
from models.stp import STPModule

class CombinedHHGNN(nn.Module):
    def __init__(self, hh_params, gnn_model, stp_params, num_edges, refractory_period=2.0, dt=0.01, synaptic_delay_steps=100):
        super().__init__()
        self.hh_node = ExtendedHodgkinHuxleyNode(**hh_params)
        self.gnn_model = gnn_model
        self.stp = STPModule(stp_params["U"], stp_params["D"], stp_params["F"])

        self.u = nn.Parameter(torch.full((num_edges,), stp_params["initial_u"], requires_grad=False))
        self.refractory_period = refractory_period
        self.synaptic_delay_steps = synaptic_delay_steps
        self.dt = dt

        # Initialize per-neuron refractory timers
        self.register_buffer("refractory_timer", torch.zeros(0))

        # Initialize synaptic delay buffer
        self.synaptic_buffer = []

    def initialize_state(self, num_nodes):
        self.refractory_timer = torch.zeros(num_nodes, device=self.u.device)
        self.synaptic_buffer = [torch.zeros(num_nodes, device=self.u.device) for _ in range(self.synaptic_delay_steps)]
        print(f"Initialized refractory timers and synaptic buffers for {num_nodes} nodes.")

    def forward(self, V, m, h, n, mCa, I_ext, data, dt, prev_spikes):
        # Update STP
        W_eff, u_new = self.stp.update(data.edge_weight, self.u, prev_spikes[data.edge_index[0]], dt)
        self.u.data = u_new.clamp(min=0.0, max=2.0)  # Clamp u to prevent it from growing indefinitely

        # Compute synaptic current (GNN)
        node_inputs = V.unsqueeze(-1)
        raw_I_syn = self.gnn_model(node_inputs, data.edge_index, W_eff).squeeze(-1)
        raw_I_syn = raw_I_syn.clamp(min=-100.0, max=100.0)  # Clamp to prevent extreme currents

        # Apply synaptic delay: push current I_syn into buffer
        self.synaptic_buffer.append(raw_I_syn)
        if len(self.synaptic_buffer) > self.synaptic_delay_steps:
            I_syn = self.synaptic_buffer.pop(0)
        else:
            I_syn = torch.zeros_like(raw_I_syn)

        # Apply refractory period
        refractory_mask = (self.refractory_timer > 0)
        effective_I_ext = I_ext.clone()
        effective_I_ext[refractory_mask] = 0.0

        # Integrate HH dynamics
        try:
            V_new, m_new, h_new, n_new, mCa_new = rk4_step_extended(self.hh_node, V, m, h, n, mCa, effective_I_ext, I_syn, dt)
        except ValueError as e:
            print(f"Integration Error: {e}")
            raise

        # Check for spikes and update refractory timers
        # A spike is detected when V crosses 0 from below
        spiked = (V <= 0) & (V_new > 0)
        self.refractory_timer[spiked] = self.refractory_period / self.dt
        # Decrement refractory timers
        self.refractory_timer = torch.clamp(self.refractory_timer - 1, min=0)

        return V_new, m_new, h_new, n_new, mCa_new, W_eff

