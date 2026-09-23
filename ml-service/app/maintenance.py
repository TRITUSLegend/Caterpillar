"""
Rule-based predictive maintenance score. No trained model - every point is traceable
to one of three explainable rules over a machine's recent telemetry window.

  Hydraulic pressure  up to 40 pts  below rated pressure and/or falling across the window
  Service interval    up to 35 pts  hours since service approaching the 500 h interval
  Fuel usage          up to 25 pts  load-adjusted fuel use rising across the window

Status: Green < 40, Amber 40-69, Red >= 70.
"""

from datetime import datetime

import numpy as np

RATED_PRESSURE_BAR = 330.0
PRESSURE_DEFICIT_START = 5.0    # bar below rated before points start
PRESSURE_DEFICIT_FULL = 35.0    # bar below rated for full points
PRESSURE_DROP_START = 3.0       # bar lost across the window before points start
PRESSURE_DROP_FULL = 15.0       # bar lost across the window for full points

SERVICE_INTERVAL_H = 500.0
SERVICE_START_H = 250.0         # hours since service before points start

IDLE_FUEL_SHARE = 0.35          # fuel burn at idle relative to working rate
FUEL_RISE_START = 0.05          # 5% rise in fuel per active hour before points start
FUEL_RISE_FULL = 0.20           # 20% rise for full points

MAX_PRESSURE, MAX_SERVICE, MAX_FUEL = 40, 35, 25
AMBER_AT, RED_AT = 40, 70


def _ramp(value, start, full):
    """0 at `start`, 1 at `full`, linear in between, clipped."""
    return float(np.clip((value - start) / (full - start), 0.0, 1.0))


def _parse(ts):
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def _window_ends(values, parts):
    """Median of the first and last 1/`parts` of the window (at least one reading each)."""
    k = max(1, len(values) // parts)
    return float(np.median(values[:k])), float(np.median(values[-k:]))


def score_machine(readings):
    """readings: list of objects with timestamp, hydraulic_pressure, fuel_used,
    idling_time, hours_since_service. Returns (score, status, reasons)."""
    readings = sorted(readings, key=lambda r: _parse(r.timestamp))
    pressure = np.array([r.hydraulic_pressure for r in readings])
    # Fuel adjusted for engine load (idling burns ~35% of working rate), so a window
    # with more idling doesn't look like wear
    active_frac = np.array([(120 - r.idling_time) / 120 for r in readings])
    fuel_rate = np.array([r.fuel_used for r in readings]) / (IDLE_FUEL_SHARE + (1 - IDLE_FUEL_SHARE) * active_frac)
    hours_since_service = readings[-1].hours_since_service

    components = []

    # Hydraulic pressure: level below rated, or loss across the window - whichever is worse
    p_start, p_end = _window_ends(pressure, parts=3)
    deficit = RATED_PRESSURE_BAR - p_end
    drop = p_start - p_end
    level_part = _ramp(deficit, PRESSURE_DEFICIT_START, PRESSURE_DEFICIT_FULL)
    trend_part = _ramp(drop, PRESSURE_DROP_START, PRESSURE_DROP_FULL)
    pressure_pts = round(MAX_PRESSURE * max(level_part, trend_part))
    if pressure_pts:
        if trend_part > level_part:
            reason = f"Hydraulic pressure down {drop:.1f} bar across window"
        else:
            reason = f"Hydraulic pressure {p_end:.1f} bar, {deficit:.1f} bar below rated {RATED_PRESSURE_BAR:.0f} bar"
        components.append((pressure_pts, reason))

    # Service interval
    service_pts = round(MAX_SERVICE * _ramp(hours_since_service, SERVICE_START_H, SERVICE_INTERVAL_H))
    if service_pts:
        components.append((
            service_pts,
            f"{hours_since_service:.0f} h since last service (interval {SERVICE_INTERVAL_H:.0f} h)",
        ))

    # Fuel usage trend
    f_start, f_end = _window_ends(fuel_rate, parts=2)
    rise = (f_end - f_start) / f_start if f_start > 0 else 0.0
    fuel_pts = round(MAX_FUEL * _ramp(rise, FUEL_RISE_START, FUEL_RISE_FULL))
    if fuel_pts:
        components.append((fuel_pts, f"Load-adjusted fuel use up {rise:.0%} across window"))

    score = min(100, sum(pts for pts, _ in components))
    status = "Red" if score >= RED_AT else "Amber" if score >= AMBER_AT else "Green"
    reasons = [] if status == "Green" else [r for _, r in sorted(components, key=lambda c: -c[0])]
    return score, status, reasons
