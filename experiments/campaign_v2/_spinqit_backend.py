r"""
_spinqit_backend.py
===================
Thin abstraction layer over the SpinQit API for the campaign v2 scripts.

Gate imports follow the SpinQit documentation:
    from spinqit import Circuit, Ry, X, CX, ...   (correct top-level import)

Provides three functions:
    run_simulator(angles, stage, ladder, shots=4096) -> np.ndarray
    run_hardware(angles, stage, ladder, conn, shots=4096) -> np.ndarray
    run_bare_hardware(conn, shots=4096) -> np.ndarray

All return a length-8 MSB-ordered probability vector:
    index k = 4*q0 + 2*q1 + q2  (q0 = most significant)

SpinQit uses little-endian keys in result.probabilities (key[0] = q0 = LSB of
the integer). The conversion to MSB is handled inside _probs_to_array().

UCRy decomposition
------------------
The 2-control UCRy on q2 uses a Mottonen-style alternating-control ladder,
verified numerically against all seven campaign distributions.

Ladder A  ctrl sequence (q1, q0, q1, q0):
    WHT input ordering [theta_00, theta_11, theta_10, theta_01]
    Gate sequence: Ry(la0), CX(q1), Ry(la1), CX(q0), Ry(la2), CX(q1), Ry(la3), CX(q0)

Ladder B  ctrl sequence (q0, q1, q0, q1):
    WHT input ordering [theta_00, theta_11, theta_01, theta_10]
    Gate sequence: Ry(la0), CX(q0), Ry(la1), CX(q1), Ry(la2), CX(q0), Ry(la3), CX(q1)

Both ladders produce identical probability distributions and differ only in
the physical pulse scheduling, as described in the companion paper.
"""

from __future__ import annotations

import time
from math import pi

import numpy as np

from spinqit import (
    Circuit,
    get_basic_simulator,
    get_compiler,
    get_nmr,
    BasicSimulatorConfig,
    NMRConfig,
    Ry, X, CX,
)


# ---------------------------------------------------------------------------
# Bit-order conversion
# ---------------------------------------------------------------------------

def _probs_to_array(probs: dict) -> np.ndarray:
    """
    Convert SpinQit result.probabilities to MSB-ordered length-8 array.

    SpinQit uses little-endian 3-bit keys: key = "b0b1b2" where b0 corresponds
    to q0 (the first allocated qubit). The MSB index is int(key, 2), which
    equals 4*b0 + 2*b1 + b2 — the same numerical value as the key interpreted
    as a binary integer. So we can convert directly.
    """
    out = np.zeros(8)
    for key, prob in probs.items():
        if len(key) == 3:
            out[int(key, 2)] = prob
    s = out.sum()
    if s > 0:
        out /= s
    return out


# ---------------------------------------------------------------------------
# GR angle computation
# ---------------------------------------------------------------------------

def gr_angles(p: np.ndarray) -> dict:
    """
    Compute all Grover-Rudolph angles for a 3-qubit probability vector.

    Parameters
    ----------
    p : np.ndarray of shape (8,), sums to 1, all entries >= 0.

    Returns
    -------
    dict with keys:
        theta0          : float  (degrees) — root rotation on q0
        theta1_0        : float  (degrees) — level-1 rotation when q0=0
        theta1_1        : float  (degrees) — level-1 rotation when q0=1
        ladder_angles_A : np.ndarray (4,) (degrees) — UCRy angles for ladder A
        ladder_angles_B : np.ndarray (4,) (degrees) — UCRy angles for ladder B
    """
    def safe_acos(num: float, den: float) -> float:
        if den < 1e-15:
            return 0.0
        return 2.0 * np.degrees(np.arccos(
            np.sqrt(np.clip(num / den, 0.0, 1.0))))

    sL  = p[0] + p[1] + p[2] + p[3]
    sR  = 1.0 - sL
    sLL = p[0] + p[1]
    sLR = p[2] + p[3]
    sRL = p[4] + p[5]
    sRR = p[6] + p[7]

    t00 = safe_acos(p[0], sLL)
    t01 = safe_acos(p[2], sLR)
    t10 = safe_acos(p[4], sRL)
    t11 = safe_acos(p[6], sRR)

    # Walsh-Hadamard transform matrix
    H = np.array([[1, 1, 1, 1],
                  [1,-1, 1,-1],
                  [1, 1,-1,-1],
                  [1,-1,-1, 1]]) / 4.0

    # Ladder A: ctrl (q1,q0,q1,q0), Gray code 00->10->11->01
    # WHT input ordering: [t00, t11, t10, t01]
    alpha_A = np.array([t00, t11, t10, t01]) / 2.0
    la_A = 2.0 * (H @ alpha_A)

    # Ladder B: ctrl (q0,q1,q0,q1), Gray code 00->01->11->10
    # WHT input ordering: [t00, t11, t01, t10]
    alpha_B = np.array([t00, t11, t01, t10]) / 2.0
    la_B = 2.0 * (H @ alpha_B)

    return {
        "theta0":          safe_acos(sL, 1.0),
        "theta1_0":        safe_acos(sLL, sL),
        "theta1_1":        safe_acos(sRL, sR),
        "ladder_angles_A": la_A,
        "ladder_angles_B": la_B,
    }


# ---------------------------------------------------------------------------
# Circuit builder
# ---------------------------------------------------------------------------

def _build_circuit(
    angles: dict,
    stage: str,
    ladder: str,
    include_measure: bool = True,
) -> Circuit:
    """
    Build a SpinQit Circuit for the given GR stage and ladder.

    Parameters
    ----------
    angles : dict — output of gr_angles()
    stage  : "L0" | "L01" | "FULL"
    ladder : "A"  | "B"
    include_measure : whether to append terminal measurements
    """
    def deg(d: float) -> float:
        return float(d) * pi / 180.0

    circ = Circuit()
    q    = circ.allocateQubits(3)
    c    = circ.allocateClbits(3) if include_measure else None

    t0   = angles["theta0"]
    t1_0 = angles["theta1_0"]
    t1_1 = angles["theta1_1"]

    # ── Level 0: Ry(theta0) on q0 ────────────────────────────────────────────
    circ << (Ry, q[0], deg(t0))

    if stage == "L0":
        if include_measure:
            circ.measure(q[0], c[0])
            circ.measure(q[1], c[1])
            circ.measure(q[2], c[2])
        return circ

    # ── Level 1: 1-control UCRy on q1 conditioned on q0 ─────────────────────
    circ << (Ry, q[1], deg(t1_0 / 2.0))
    circ << (X,  q[0])
    circ << (CX, (q[0], q[1]))
    circ << (Ry, q[1], deg(720.0 - t1_0 / 2.0))
    circ << (CX, (q[0], q[1]))
    circ << (X,  q[0])
    circ << (Ry, q[1], deg(t1_1 / 2.0))
    circ << (CX, (q[0], q[1]))
    circ << (Ry, q[1], deg(720.0 - t1_1 / 2.0))
    circ << (CX, (q[0], q[1]))

    if stage == "L01":
        if include_measure:
            circ.measure(q[0], c[0])
            circ.measure(q[1], c[1])
            circ.measure(q[2], c[2])
        return circ

    # ── Level 2: 2-control UCRy on q2 — ladders A and B ─────────────────────
    if ladder == "A":
        # ctrl sequence (q1, q0, q1, q0)
        # WHT ordering [t00, t11, t10, t01]
        la = [deg(a) for a in angles["ladder_angles_A"]]
        circ << (Ry, q[2], la[0])
        circ << (CX, (q[1], q[2]))
        circ << (Ry, q[2], la[1])
        circ << (CX, (q[0], q[2]))
        circ << (Ry, q[2], la[2])
        circ << (CX, (q[1], q[2]))
        circ << (Ry, q[2], la[3])
        circ << (CX, (q[0], q[2]))
    else:  # ladder B
        # ctrl sequence (q0, q1, q0, q1)
        # WHT ordering [t00, t11, t01, t10]
        la = [deg(a) for a in angles["ladder_angles_B"]]
        circ << (Ry, q[2], la[0])
        circ << (CX, (q[0], q[2]))
        circ << (Ry, q[2], la[1])
        circ << (CX, (q[1], q[2]))
        circ << (Ry, q[2], la[2])
        circ << (CX, (q[0], q[2]))
        circ << (Ry, q[2], la[3])
        circ << (CX, (q[1], q[2]))

    if include_measure:
        circ.measure(q[0], c[0])
        circ.measure(q[1], c[1])
        circ.measure(q[2], c[2])

    return circ


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_simulator(angles: dict, stage: str, ladder: str = "A",
                  shots: int = 4096) -> np.ndarray:
    """Run ideal SpinQit BasicSimulator. Returns MSB-ordered prob vector."""
    circ   = _build_circuit(angles, stage, ladder, include_measure=True)
    comp   = get_compiler("native")
    exe    = comp.compile(circ, 0)
    engine = get_basic_simulator()
    config = BasicSimulatorConfig()
    config.configure_shots(shots)
    result = engine.execute(exe, config)
    return _probs_to_array(result.probabilities)


def run_hardware(angles: dict, stage: str, ladder: str,
                 conn: dict, shots: int = 4096,
                 retries: int = 3, retry_wait: float = 10.0) -> np.ndarray:
    """
    Run on SpinQ Triangulum NMR via SpinQit NMR backend.
    Returns MSB-ordered prob vector.

    conn keys: ip, port, account, password,
               task_name (optional), task_desc (optional)
    """
    circ   = _build_circuit(angles, stage, ladder, include_measure=False)
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
                conn.get("task_desc", "GR characterisation v2"),
            )
            config.configure_shots(shots)
            result = engine.execute(exe, config)
            return _probs_to_array(result.probabilities)
        except Exception as exc:
            last_exc = exc
            if attempt < retries:
                time.sleep(retry_wait)

    raise RuntimeError(
        f"Hardware execution failed after {retries} attempts: {last_exc}"
    )


def run_bare_hardware(conn: dict, shots: int = 4096) -> np.ndarray:
    """
    Bare-reference hardware check for the NMR backend.

    Important:
    - Triangulum NMR does not accept explicit MEASURE operations.
    - Some SpinQit/NMR versions also reject a completely empty circuit.
    Therefore we use a non-empty identity circuit: X followed by X on q0.
    """
    circ = Circuit()
    q    = circ.allocateQubits(3)

    # Non-empty identity circuit to avoid empty-graph compilation issues.
    circ << (X, q[0])
    circ << (X, q[0])

    comp   = get_compiler("native")
    exe    = comp.compile(circ, 0)
    engine = get_nmr()

    config = NMRConfig()
    config.configure_ip(conn["ip"])
    config.configure_port(int(conn["port"]))
    config.configure_account(conn["account"], conn["password"])
    config.configure_task(
        conn.get("task_name", "bare_check"),
        conn.get("task_desc", "Pre-session bare-state check"),
    )
    config.configure_shots(shots)
    result = engine.execute(exe, config)
    return _probs_to_array(result.probabilities)