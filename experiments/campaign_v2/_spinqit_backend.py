r"""
_spinqit_backend.py
===================
Thin abstraction layer over the SpinQit API used by the campaign v2 scripts.

Provides two functions with a uniform interface:

    run_simulator(angles, stage, ladder, shots) -> np.ndarray
    run_hardware(angles, stage, ladder, conn, shots) -> np.ndarray

Both return a length-8 numpy probability vector ordered MSB->LSB:
    index 0 = |000>, 1 = |001>, ..., 7 = |111>

SpinQit uses little-endian ordering in result.probabilities
(first bit = first qubit = LSB), so results are reordered here
to match the MSB convention used throughout this campaign.

References
----------
SpinQit documentation: https://doc.spinq.cn/doc/spinqit/index.html
"""

from __future__ import annotations

import time
from math import pi
from typing import TYPE_CHECKING

import numpy as np

from spinqit import (
    Circuit,
    get_basic_simulator,
    get_compiler,
    get_nmr,
    BasicSimulatorConfig,
    NMRConfig,
)
from spinqit.primitive import Ry, X, CX

# ── State ordering ────────────────────────────────────────────────────────────
# SpinQit little-endian keys: "000" means q0=0, q1=0, q2=0 but the string
# is written q2q1q0. We need MSB ordering: key "abc" -> q0=a, q1=b, q2=c.
# Map: spinqit key (LSB first) -> MSB index
_SPINQIT_KEYS = [f"{i:03b}" for i in range(8)]                  # 000..111 LSB
_MSB_KEYS     = [k[::-1] for k in _SPINQIT_KEYS]                # reversed to MSB
_LSB_TO_MSB   = {lsb: int(msb, 2)
                 for lsb, msb in zip(_SPINQIT_KEYS, _MSB_KEYS)} # lsb str -> msb idx


def _probs_dict_to_array(probs: dict) -> np.ndarray:
    """Convert SpinQit result.probabilities dict to MSB-ordered length-8 array."""
    out = np.zeros(8)
    for lsb_key, prob in probs.items():
        msb_idx = _LSB_TO_MSB.get(lsb_key, None)
        if msb_idx is not None:
            out[msb_idx] = prob
    # Normalise (floating-point rounding from SpinQit)
    s = out.sum()
    if s > 0:
        out /= s
    return out


# ── Circuit builder ───────────────────────────────────────────────────────────

def _build_circuit(angles: dict, stage: str, ladder: str) -> Circuit:
    """
    Build a SpinQit Circuit for the given GR stage and ladder.

    Parameters
    ----------
    angles : dict
        Output of gr_angles() from 00_generate_distributions.py.
        Keys used:
            theta0          float  (degrees)
            theta1_0        float  (degrees)
            theta1_1        float  (degrees)
            ladder_angles   np.ndarray shape (4,)  (degrees, WHT result * 2)
    stage  : "L0" | "L01" | "FULL"
    ladder : "A" | "B"
    """
    def deg2rad(d: float) -> float:
        return float(d) * pi / 180.0

    circ = Circuit()
    q    = circ.allocateQubits(3)   # q[0], q[1], q[2]

    t0   = deg2rad(angles["theta0"])
    t1_0 = deg2rad(angles["theta1_0"])
    t1_1 = deg2rad(angles["theta1_1"])
    phi  = [deg2rad(a) for a in angles["ladder_angles"]]  # 4 values

    # ── Level 0: single Ry on q0 ─────────────────────────────────────────────
    circ << (Ry, q[0], t0)

    if stage == "L0":
        circ.measure(q[0], circ.allocateClbits(1)[0])
        circ.measure(q[1], circ.allocateClbits(1)[0])
        circ.measure(q[2], circ.allocateClbits(1)[0])
        return circ

    # ── Level 1: UCRy on q1 controlled by q0 (Gray-code 1-ctrl decomp) ──────
    # CRy(theta1_0) when q0=0 : implemented as X(q0), CRy, X(q0)
    # CRy(theta1_1) when q0=1 : plain CRy
    #
    # 1-ctrl UCRy decomposition (Mottonen / ladder A=B at level 1):
    #   Ry(+a) on q1, CNOT(q0->q1), Ry(-b) on q1, CNOT(q0->q1)
    # where a = (t1_0 + t1_1)/2 / 2,  b = (t1_0 - t1_1)/2 / 2
    # (the /2 factors come from the WHT for 1 control)
    a1 = (t1_0 + t1_1) / 4.0
    b1 = (t1_0 - t1_1) / 4.0

    circ << (Ry,  q[1],  a1)
    circ << (X,   q[0])
    circ << (CX,  q[0],  q[1])
    circ << (Ry,  q[1], -b1)
    circ << (X,   q[0])
    circ << (CX,  q[0],  q[1])
    circ << (Ry,  q[1],  b1)
    circ << (CX,  q[0],  q[1])
    circ << (Ry,  q[1], -a1)
    circ << (CX,  q[0],  q[1])

    if stage == "L01":
        circ.measure(q[0], circ.allocateClbits(1)[0])
        circ.measure(q[1], circ.allocateClbits(1)[0])
        circ.measure(q[2], circ.allocateClbits(1)[0])
        return circ

    # ── Level 2: 2-ctrl UCRy on q2, ladders A and B ──────────────────────────
    # phi[0..3] are the WHT-transformed half-angles (already in radians).
    # Ladder A control sequence: q1, q0, q0, q0
    # Ladder B control sequence: q1, q1, q0, q0
    #
    # Each ladder step: Ry(2*phi[i]) on q2, then CNOT(ctrl -> q2)

    circ << (Ry, q[2], 2 * phi[0])

    if ladder == "A":
        circ << (CX, q[1], q[2])
        circ << (Ry, q[2], 2 * phi[3])
        circ << (CX, q[0], q[2])
        circ << (Ry, q[2], 2 * phi[2])
        circ << (CX, q[0], q[2])
        circ << (Ry, q[2], 2 * phi[1])
        circ << (CX, q[0], q[2])
    else:  # ladder B
        circ << (CX, q[1], q[2])
        circ << (Ry, q[2], 2 * phi[1])
        circ << (CX, q[1], q[2])
        circ << (Ry, q[2], 2 * phi[2])
        circ << (CX, q[0], q[2])
        circ << (Ry, q[2], 2 * phi[3])
        circ << (CX, q[0], q[2])

    circ.measure(q[0], circ.allocateClbits(1)[0])
    circ.measure(q[1], circ.allocateClbits(1)[0])
    circ.measure(q[2], circ.allocateClbits(1)[0])
    return circ


# ── Public API ────────────────────────────────────────────────────────────────

def run_simulator(angles: dict, stage: str, ladder: str,
                  shots: int = 4096) -> np.ndarray:
    """
    Run the ideal SpinQit BasicSimulator for the given GR circuit.
    Returns a length-8 MSB-ordered probability vector.
    """
    circ   = _build_circuit(angles, stage, ladder)
    comp   = get_compiler("native")
    exe    = comp.compile(circ, 0)
    engine = get_basic_simulator()
    config = BasicSimulatorConfig()
    config.configure_shots(shots)
    result = engine.execute(exe, config)
    return _probs_dict_to_array(result.probabilities)


def run_hardware(angles: dict, stage: str, ladder: str,
                 conn: dict, shots: int = 4096,
                 retries: int = 3, retry_wait: float = 10.0) -> np.ndarray:
    """
    Run on the SpinQ Triangulum NMR hardware via SpinQit NMR backend.
    conn must contain: ip, port, account, password.
    Returns a length-8 MSB-ordered probability vector.
    Retries up to `retries` times on hardware error.
    """
    circ   = _build_circuit(angles, stage, ladder)
    comp   = get_compiler("native")
    exe    = comp.compile(circ, 0)
    engine = get_nmr()

    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            config = NMRConfig()
            config.configure_ip(conn["ip"])
            config.configure_port(int(conn["port"]))
            config.configure_account(conn["account"], conn["password"])
            config.configure_task(
                conn.get("task_name", "gr_campaign_v2"),
                conn.get("task_desc", "GR characterisation campaign v2"),
            )
            config.configure_shots(shots)
            result = engine.execute(exe, config)
            return _probs_dict_to_array(result.probabilities)
        except Exception as exc:
            last_exc = exc
            if attempt < retries:
                time.sleep(retry_wait)

    raise RuntimeError(
        f"Hardware execution failed after {retries} attempts: {last_exc}"
    )


def run_bare_hardware(conn: dict, shots: int = 4096) -> np.ndarray:
    """
    Run a bare measurement (identity circuit) on hardware.
    Used for the pre-session bare-state check.
    Returns a length-8 MSB-ordered probability vector.
    """
    circ = Circuit()
    q    = circ.allocateQubits(3)
    c    = circ.allocateClbits(3)
    circ.measure(q[0], c[0])
    circ.measure(q[1], c[1])
    circ.measure(q[2], c[2])

    comp   = get_compiler("native")
    exe    = comp.compile(circ, 0)
    engine = get_nmr()

    config = NMRConfig()
    config.configure_ip(conn["ip"])
    config.configure_port(int(conn["port"]))
    config.configure_account(conn["account"], conn["password"])
    config.configure_task(
        conn.get("task_name", "bare_check"),
        conn.get("task_desc", "Bare state check"),
    )
    config.configure_shots(shots)
    result = engine.execute(exe, config)
    return _probs_dict_to_array(result.probabilities)
