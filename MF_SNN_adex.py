from pyrates import CircuitTemplate
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import cauchy
import yaml
from rectipy import Network, random_connectivity
plt.rcParams['backend'] = 'TkAgg'

### defining probability distribution
def lorentzian(n: int, eta: float, delta: float, lb: float, ub: float):
    samples = np.zeros((n,))
    for i in range(n):
        s = cauchy.rvs(loc=eta, scale=delta)
        while s <= lb or s >= ub:
            s = cauchy.rvs(loc=eta, scale=delta)
        samples[i] = s
    return samples

### defining parameters

# model parameters
N = 1000
p = 0.2
C = 100.0
k = 0.7
v_r = -60.0
v_t = -40.0
Delta = 2.5
eta = 0.0
a = 0.03
b = -2.0
d = 10.0
g = 15.0
E_r = 0.0
tau_s = 6.0
v_spike = 1000.0
v_reset = -1000.0
# Add additional parameters for the AdEx model
C_adex = 281.0
delta_T = 2.0
g_adex = 30.0 
a_adex = 4.0
v_r_adex = -70.6
b_adex = 80.5
tau_w = 144.0

# define inputs
T = 2500.0
cutoff = 500.0
dt = 1e-2
dts = 1e-1
inp = np.zeros((int(T/dt), 1)) + 25.0
inp[:int(cutoff*0.5/dt), 0] -= 15.0
inp[int(750/dt):int(2000/dt), 0] += 30

# define lorentzian of etas
thetas = lorentzian(N, eta=v_t, delta=Delta, lb=v_r, ub=0.0)

# define connectivity
W = random_connectivity(N, N, p, normalize=True)


##############################################

# run the mean-field model

# initialize model
adex = CircuitTemplate.from_yaml("/Users/utilizator/Desktop/gast_paper/config/adex_mf/adex")

# update parameters --- TODO!
adex.update_var(node_vars={'p/adex_op/C': C_adex,  'p/adex_op/v_r': v_r_adex, 'p/adex_op/v_t': v_t, 'p/adex_op/delta_T': delta_T,
                         'p/adex_op/a': a_adex, 'p/adex_op/b': b_adex, 'p/adex_op/tau_s': tau_s, 'p/adex_op/g': g_adex,
                         'p/adex_op/E_r': E_r, 'p/adex_op/tau_w': tau_w, 'p/adex_op/delta_v': Delta})

# run simulation
res_mf_adex = adex.run(simulation_time=T, step_size=dt, sampling_step_size=dts, cutoff=cutoff, solver='euler',
                outputs={'s': 'p/adex_op/s'}, inputs={'p/adex_op/I_ext': inp[:, 0]})


# run SNN simulation

## ADEX VERSION
node_vars_adex = { "eta": eta, "v_r": v_r, "delta_T" : delta_T,  "v_theta": thetas,  "E_r": E_r, 
                  "C": C_adex, "a": a_adex, "b": b_adex, "tau_w": tau_w, "g": g, "tau_s": tau_s, "v" : v_t}


# initialize model
net = Network(dt=dt, device="cpu")  # we need to change all the node vars etc for the adex model
net.add_diffeq_node("snn", "/Users/utilizator/Desktop/gast_paper/config/adex_snn/rs", weights=W, source_var="s", target_var="s_in",
                    input_var="I_ext", output_var="s", spike_var="spike", reset_var="v", to_file=False,
                    node_vars=node_vars_adex.copy(), op="rs_op", spike_reset=v_reset, spike_threshold=v_spike,
                    verbose=False, clear=True, device="cuda:0")

# perform simulation
obs = net.run(inputs=inp, sampling_steps=int(dts/dt), record_output=True, verbose=True, cutoff=int(cutoff/dt))
res_snn_adex = obs.to_dataframe("out")

# plot results
t = res_mf_adex.index
plt.figure(figsize=(10, 5))
plt.plot(t, res_mf_adex["s"], label="Mean-field")
plt.plot(t, np.mean(res_snn_adex, axis=1), label="SNN")
plt.title("AdEx model - Mean-field vs. SNN")
plt.xlabel("Time [ms]")
plt.ylabel("Activity")
plt.legend()