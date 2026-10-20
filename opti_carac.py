import jax
import jax.numpy as jnp
from dataclasses import dataclass
from jax.tree_util import register_dataclass
import scipy.optimize as scio
import matplotlib.pyplot as plt 
from matplotlib import ticker

from post_process_parquet import calculate_dataframe

import plotly.graph_objects as go

jax.config.update("jax_enable_x64", True) 
jax.config.update("jax_debug_nans", True)
cmap = 'jet'

# data_coupled = calculate_dataframe("2026_10_16_18h29")
data_uncoupled = calculate_dataframe("2026_9_23_18h19")

# Récupère les listes de fréquences et de résistances et jnp array
frequency_list = jnp.asarray(data_uncoupled['f'])
resistance_list = jnp.asarray(data_uncoupled['Rload'])
power1_exp = jnp.asarray(data_uncoupled['pow1'])
power2_exp = jnp.asarray(data_uncoupled['pow2'])

# class SystemParams:
@register_dataclass  # rend la classe jax-compatible
@dataclass
class SystemParams:
    gamma: float = (0.1,)
    w: float = (100 * 2 * jnp.pi,)
    R: float = (10e3,)

def make_proto_carac(w0, cp, km2, Q, M):
    return jnp.array([w0, cp, km2, Q, M])

ProtoCaracsRef1 = make_proto_carac(95 * 2 * jnp.pi, 5.7781e-9, 0.35342, 65.701, 0.05)
ProtoCaracsRef2 = make_proto_carac(127.96 * 2 * jnp.pi, 3.7888e-9, 1.1055, 66.846, 0.05)

# Fonction qui calcule la puissance
def system_signals(params: SystemParams, vect):
    """
    returns x1, x2, phi_x1, phi_x2, v1, v2, P1in, P2in
    position resonator 1, voltage resonator 1
    position resonator 2, voltage resonator 2
    computed by analytical analysis
    for the two couples linear resonators system
    """

    gamma = params.gamma
    w = params.w
    R = params.R

    w0 = vect[0]
    Cp = vect[1]
    km2 = vect[2]
    Q = vect[3]
    M = vect[4]

    alpha = jnp.sqrt(km2 * M * w0**2 * Cp)
    a = -(w**2) + 1.0j * w * w0 / Q + w0**2
    b = alpha / M
    c = 1.0j * w * alpha
    d = 1.0j * w * Cp + 1 / R

    v = c * gamma / (b * c + d * a)

    P = jnp.abs(v) ** 2 / (2 * R)

    return w, R, P


# Calcul de la puissance en vmap sur omega et R
vpower = jax.vmap(
    jax.vmap(
        system_signals,
        in_axes=(
            SystemParams(
                gamma=None,
                w=0,
                R=None,
            ),
            None,
        ),
    ),
    in_axes=(
        SystemParams(
            gamma=None,
            w=None,
            R=0,
        ),
        None,
    ),
)

def power(vect):
    params_w = SystemParams(
        gamma=0.1,
        w=frequency_list * 2 * jnp.pi,
        R=resistance_list,
    )

    return vpower(params_w, vect)

def error1_function(X0):
    caracs = X0 * ProtoCaracsRef1
    w, r, power_sim = power(caracs)
    return (power_sim - power1_exp).flatten()

def error2_function(X0):
    caracs = X0 * ProtoCaracsRef2
    w, r, power_sim = power(caracs)
    return (power_sim - power2_exp).flatten()

def find_carac_opt(proto):
    x0 = jnp.ones_like(proto)
    res = scio.least_squares(error1_function, x0)
    xf = res.x
    return xf * proto


ProtoCaracsOpt1 = find_carac_opt(ProtoCaracsRef1)
print(
    f"w01 = {ProtoCaracsOpt1[0] / (2 * jnp.pi)}, \nCp1 = {ProtoCaracsOpt1[1]}, \nkm21 = {ProtoCaracsOpt1[2]}, \nQ1 = {ProtoCaracsOpt1[3]}, \nM1 = {ProtoCaracsOpt1[4]}"
)