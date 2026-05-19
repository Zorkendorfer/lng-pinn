"""FSRU steady-state regasification model (ground truth).

Equipment topology (Independence FSRU, simplified):
  1. Cryogenic pump: raises LNG from storage pressure (~1 bar) to send-out pressure.
  2. Open-rack vaporizer (ORV): seawater-heated shell-and-tube, vaporises LNG.
  3. Trim heater: electric resistance, raises gas to send-out temperature.

All equations are steady-state energy and mass balances; no transient terms.
"""

from __future__ import annotations

from dataclasses import dataclass

import CoolProp.CoolProp as CP

from lng_pinn.thermo import get_state

# Plant constants (Independence FSRU reference values)
ETA_PUMP_BEP = 0.78        # peak isentropic pump efficiency (at BEP)
M_DOT_BEP = 45.0           # kg/s - best-efficiency-point flow rate
ETA_PUMP_CURVATURE = 8e-5  # (kg/s)^-2 - quadratic curvature around BEP
# B2 (trim heater turndown penalty) explicitly skipped for v1.1 - candidate for v2.
ETA_TRIM_HEATER = 0.98  # trim heater thermal efficiency
P_IN = 1.0e5  # Pa - LNG storage pressure (atmospheric after boil-off)
T_IN = 111.0  # K  - LNG storage temperature (bubble point at 1 bar, approx)
T_SENDOUT = 278.15  # K  - target send-out temperature (5  deg C)
P_OUT_DEFAULT = 80.0e5  # Pa - send-out pressure (80 bar)
J_TO_KWH = 1.0 / 3_600_000.0  # conversion factor


def pump_efficiency(m_dot: float) -> float:
    """Quadratic pump efficiency curve centred on the best-efficiency point.

    Conservative literature values: large cryogenic pumps lose ~10-15
    efficiency points at 25% and 150% of BEP. See Karassik, Pump Handbook section 2.
    """
    return ETA_PUMP_BEP - ETA_PUMP_CURVATURE * (m_dot - M_DOT_BEP) ** 2


@dataclass
class PlantOutput:
    W_pump: float  # kWh / kg - pump electrical work
    W_trim: float  # kWh / kg - trim heater electrical work
    W_total: float  # kWh / kg - total electrical energy
    T_out: float  # K  - actual send-out gas temperature
    Q_sw: float  # kWh / kg - seawater heat duty (ORV)
    exergy_destruction: float  # kWh / kg - exergy destroyed in vaporiser


def simulate(
    composition: tuple[float, ...],
    m_dot: float,
    T_amb: float,
    T_sw: float,
    P_out: float = P_OUT_DEFAULT,
) -> PlantOutput:
    """Simulate steady-state regasification for one operating point.

    Args:
        composition: mole fractions (CH4, C2H6, C3H8, nC4, iC4, N2), summing to 1.
        m_dot:       send-out mass flow rate, kg/s.
        T_amb:       ambient air temperature, K (unused in ORV topology but kept for API).
        T_sw:        seawater temperature, K.
        P_out:       send-out pressure, Pa.

    Returns:
        PlantOutput with energy quantities normalised per kg of gas sent out.
    """
    x = tuple(composition)
    state = get_state(x)

    # --- Inlet state (saturated liquid) ---
    # specify_phase avoids solver divergence when the singleton was previously at high P.
    state.specify_phase(CP.iphase_liquid)
    state.update(CP.PT_INPUTS, P_IN, T_IN)
    h_in = state.hmolar()    # J/mol
    s_in = state.smolar()    # J/(mol*K)
    rho_in = state.rhomass() # kg/m^3
    mw = state.molar_mass()  # kg/mol
    state.unspecify_phase()

    # --- Pump: isentropic work, corrected for flow-dependent efficiency ---
    # Approximation: liquid is incompressible, v ~ 1/rho_in
    v_liq = 1.0 / rho_in                         # m^3/kg
    w_pump_is = v_liq * (P_out - P_IN)           # J/kg (isentropic)
    eta_p = pump_efficiency(m_dot)
    w_pump = w_pump_is / eta_p                   # J/kg (actual)
    h_after_pump = h_in + w_pump * mw            # J/mol

    # --- Target state at send-out conditions ---
    state.update(CP.PT_INPUTS, P_out, T_SENDOUT)
    h_out_target = state.hmolar()  # J/mol

    # --- ORV outlet: seawater approach temperature ---
    T_orv_out = min(T_sw - 3.0, T_SENDOUT)
    state.update(CP.PT_INPUTS, P_out, T_orv_out)
    h_orv_out = state.hmolar()   # J/mol
    s_orv_out = state.smolar()   # J/(mol*K)
    cp_orv = state.cpmass()      # J/(kg*K)

    q_orv_actual = max(0.0, h_orv_out - h_after_pump)  # J/mol
    q_sw_kg = q_orv_actual / mw                         # J/kg

    # Trim heater
    q_trim = max(0.0, h_out_target - h_orv_out)  # J/mol
    w_trim = (q_trim / mw) / ETA_TRIM_HEATER      # J/kg

    # --- Actual outlet temperature ---
    T_out = (
        T_SENDOUT
        if T_orv_out >= T_SENDOUT
        else T_orv_out + (q_trim / mw) / (cp_orv or 2200.0)
    )

    # --- Exergy destruction in ORV ---
    # Exergy supplied by seawater: Q_sw * (1 - T0/T_sw)
    # Exergy gained by stream: (h_out - h_in) - T0*(s_out - s_in)  [per kg]
    T0 = 273.15
    exergy_in_sw = q_sw_kg * (1.0 - T0 / T_sw) if T_sw > T0 else 0.0
    exergy_stream_gain = (h_orv_out - h_in) / mw - T0 * (s_orv_out - s_in) / mw
    exergy_destruction = max(0.0, exergy_in_sw - exergy_stream_gain)

    return PlantOutput(
        W_pump=w_pump * J_TO_KWH,
        W_trim=w_trim * J_TO_KWH,
        W_total=(w_pump + w_trim) * J_TO_KWH,
        T_out=T_out,
        Q_sw=q_sw_kg * J_TO_KWH,
        exergy_destruction=exergy_destruction * J_TO_KWH,
    )
