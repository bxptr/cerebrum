import torch

class ExtendedHodgkinHuxleyNode(torch.nn.Module):
    def __init__(self, gNa=120.0, gK=36.0, gL=0.3, gCa=1.0, ENa=50.0, EK=-77.0, EL=-54.4, ECa=120.0, Cm=1.0):
        super().__init__()
        self.gNa = gNa
        self.gK = gK
        self.gL = gL
        self.gCa = gCa
        self.ENa = ENa
        self.EK = EK
        self.EL = EL
        self.ECa = ECa
        self.Cm = Cm

    def alpha_m(self, V):
        return torch.where(
            V != -40,
            0.1 * (V + 40) / (1 - torch.exp(-(V + 40) / 10)),
            1.0  # Limit as V approaches -40
        )

    def beta_m(self, V):
        return 4.0 * torch.exp(-(V + 65) / 18)

    def alpha_h(self, V):
        return 0.07 * torch.exp(-(V + 65) / 20)

    def beta_h(self, V):
        return 1 / (1 + torch.exp(-(V + 35) / 10))

    def alpha_n(self, V):
        return torch.where(
            V != -55,
            0.01 * (V + 55) / (1 - torch.exp(-(V + 55) / 10)),
            0.1  # Limit as V approaches -55
        )

    def beta_n(self, V):
        return 0.125 * torch.exp(-(V + 65) / 80)

    def mCa_inf(self, V):
        return 1 / (1 + torch.exp(-(V + 20) / 9))

    def tau_mCa(self, V):
        return 5.0

    def forward(self, V, m, h, n, mCa, I_ext, I_syn):
        # Clamp V to prevent numerical issues
        V = torch.clamp(V, min=-100.0, max=100.0)

        INa = self.gNa * m**3 * h * (V - self.ENa)
        IK = self.gK * n**4 * (V - self.EK)
        IL = self.gL * (V - self.EL)
        
        mCa_inf_val = self.mCa_inf(V)
        mCa_tau = self.tau_mCa(V)
        dmCa_dt = (mCa_inf_val - mCa) / mCa_tau
        ICa = self.gCa * mCa * (V - self.ECa)

        dVdt = (I_ext - (INa + IK + IL + ICa) + I_syn) / self.Cm
        dmdt = self.alpha_m(V) * (1 - m) - self.beta_m(V) * m
        dhdt = self.alpha_h(V) * (1 - h) - self.beta_h(V) * h
        dndt = self.alpha_n(V) * (1 - n) - self.beta_n(V) * n

        return dVdt, dmdt, dhdt, dndt, dmCa_dt

