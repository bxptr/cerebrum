import torch

def rk4_step_extended(model, V, m, h, n, mCa, I_ext, I_syn, dt):
    def f(V, m, h, n, mCa):
        return model(V, m, h, n, mCa, I_ext, I_syn)

    k1 = f(V, m, h, n, mCa)
    k2 = f(V + k1[0]*dt/2, m + k1[1]*dt/2, h + k1[2]*dt/2, n + k1[3]*dt/2, mCa + k1[4]*dt/2)
    k3 = f(V + k2[0]*dt/2, m + k2[1]*dt/2, h + k2[2]*dt/2, n + k2[3]*dt/2, mCa + k2[4]*dt/2)
    k4 = f(V + k3[0]*dt,   m + k3[1]*dt,   h + k3[2]*dt,   n + k3[3]*dt,   mCa + k3[4]*dt)

    V_new = V + (dt/6)*(k1[0] + 2*k2[0] + 2*k3[0] + k4[0])
    m_new = m + (dt/6)*(k1[1] + 2*k2[1] + 2*k3[1] + k4[1])
    h_new = h + (dt/6)*(k1[2] + 2*k2[2] + 2*k3[2] + k4[2])
    n_new = n + (dt/6)*(k1[3] + 2*k2[3] + 2*k3[3] + k4[3])
    mCa_new = mCa + (dt/6)*(k1[4] + 2*k2[4] + 2*k3[4] + k4[4])

    # Clamp V_new to prevent explosion
    V_new = torch.clamp(V_new, min=-100.0, max=100.0)

    # Ensure no NaNs are produced
    if not torch.isfinite(V_new).all():
        raise ValueError("V_new contains non-finite values after RK4 step.")
    if not torch.isfinite(m_new).all():
        raise ValueError("m_new contains non-finite values after RK4 step.")
    if not torch.isfinite(h_new).all():
        raise ValueError("h_new contains non-finite values after RK4 step.")
    if not torch.isfinite(n_new).all():
        raise ValueError("n_new contains non-finite values after RK4 step.")
    if not torch.isfinite(mCa_new).all():
        raise ValueError("mCa_new contains non-finite values after RK4 step.")

    return V_new, m_new, h_new, n_new, mCa_new

