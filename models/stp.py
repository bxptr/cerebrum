import torch

class STPModule(torch.nn.Module):
    """
    Short-Term Plasticity (STP) model for synaptic weights.
    Each synapse has a use-dependent variable 'u' and the effective weight changes with activity.
    We implement a simplified model:
    u(t): usage parameter that increases with spikes (facilitation) and decays back
    W_effective = W * u
    Depression and facilitation are governed by time constants D and F.
    Reference: Markram et al. (1998).
    """
    def __init__(self, U=0.2, D=200.0, F=150.0):
        super().__init__()
        self.U = U
        self.D = D
        self.F = F

    def update(self, edge_weight, u, spikes, dt):
        # spikes: binary or continuous measure of activity on the pre-synaptic neuron
        # u dynamics: du = ((U - u)/F + U*(1-u)*spikes) * dt
        # When a spike occurs, u increases; otherwise, it recovers back to U.
        du = ((self.U - u)/self.F) + self.U*(1 - u)*spikes
        u_new = u + du*dt
        u_new = torch.clamp(u_new, min=0.0, max=1.0)  # Clamp u to [0,1]

        # Effective weight
        W_eff = edge_weight * u_new
        return W_eff, u_new

