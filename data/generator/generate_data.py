"""
Synthetic data generator for the Smart Operator Assistant.

Produces three CSVs matching the final report schema:
  operators.csv  - Operator_ID, Name, Role, Skill_Level
  tasks.csv      - Task_ID, Operator_ID, Machine_ID, Task_Type, Weather, Site_Terrain,
                   Operator_Skill, Machine_Age, Estimated_Time, Actual_Time
  telemetry.csv  - Timestamp, Machine_ID, Operator_ID, Shift_ID, Machine_Type, GPS_Zone,
                   Engine_Hours, Fuel_Used, Load_Cycles, Idling_Time, Hydraulic_Pressure,
                   Hours_Since_Service, Seatbelt_Status, Proximity_Alert, Safety_Alert_Triggered

Built-in patterns:
  - Hydraulic_Pressure drifts down and Fuel_Used rises with Hours_Since_Service (wear).
    M03 degrades fastest; M05 gets serviced mid-period (reset + recovery).
  - Rainy weather + Beginner operator -> much larger Actual_Time vs Estimated_Time gap.
  - Idling_Time spikes cluster in the last readings of each shift.
  - Proximity_Alert = another machine in the same GPS_Zone at the same Timestamp.
  - Safety_Alert_Triggered = seatbelt unfastened OR proximity alert.

Usage (local):  python data/generator/generate_data.py
Usage (Colab):  !python generate_data.py --out-dir .
"""

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

START_DATE = datetime(2026, 9, 1, tzinfo=timezone.utc)
NUM_DAYS = 18
NUM_TASKS = 1000
MIN_TASKS_PER_SLOT = 3

# Shift A 06:00-14:00, Shift B 14:00-22:00. One reading every 2h, mid-interval.
SHIFTS = {
    "A": [7, 9, 11, 13],
    "B": [15, 17, 19, 21],
}
READING_INTERVAL_H = 2.0

ZONES = [f"Zone_{c}" for c in "ABCDEFGH"]
WEATHER = ["Clear", "Cloudy", "Rainy", "Windy"]
WEATHER_P = [0.45, 0.25, 0.20, 0.10]
TERRAIN = ["Flat", "Rocky", "Muddy", "Sloped"]

MACHINES = {
    # id: type, age in years at START_DATE, home zone, base pressure (bar),
    #     wear (bar lost per hour since service), hours since service at START_DATE
    "M01": dict(type="Excavator",         age=3.2, zone="Zone_A", pressure=332, wear=0.020, hss=120),
    "M02": dict(type="Wheel Loader",      age=5.5, zone="Zone_E", pressure=328, wear=0.025, hss=60),
    "M03": dict(type="Excavator",         age=8.1, zone="Zone_B", pressure=330, wear=0.080, hss=220),
    "M04": dict(type="Dozer",             age=4.0, zone="Zone_C", pressure=335, wear=0.020, hss=150),
    "M05": dict(type="Articulated Truck", age=6.7, zone="Zone_D", pressure=326, wear=0.050, hss=380),
    "M06": dict(type="Motor Grader",      age=2.4, zone="Zone_F", pressure=331, wear=0.015, hss=90),
}
SERVICE_EVENTS = {"M05": 9}  # machine -> day index serviced (before shift A)

FUEL_RATE_LPH = {
    "Excavator": 18, "Wheel Loader": 15, "Dozer": 22,
    "Motor Grader": 12, "Articulated Truck": 20,
}
LOAD_CYCLES_PER_INTERVAL = {
    "Excavator": 60, "Wheel Loader": 45, "Dozer": 25,
    "Motor Grader": 10, "Articulated Truck": 6,
}
TASK_TYPES_BY_MACHINE = {
    "Excavator": ["Excavation", "Trenching", "Loading"],
    "Wheel Loader": ["Loading", "Hauling"],
    "Dozer": ["Grading", "Excavation"],
    "Motor Grader": ["Grading"],
    "Articulated Truck": ["Hauling"],
}
TASK_BASE_MINUTES = {
    "Excavation": 65, "Trenching": 60, "Loading": 35, "Hauling": 45, "Grading": 55,
}

SKILL_EFFECT = {"Beginner": 0.15, "Intermediate": 0.05, "Expert": -0.05}
WEATHER_EFFECT = {"Clear": 0.0, "Cloudy": 0.02, "Rainy": 0.15, "Windy": 0.07}
TERRAIN_EFFECT = {"Flat": 0.0, "Rocky": 0.08, "Muddy": 0.12, "Sloped": 0.06}
RAINY_BEGINNER_EXTRA = 0.25
SEATBELT_UNFASTENED_P = {"Beginner": 0.06, "Intermediate": 0.03, "Expert": 0.01}

ANOMALY_RATE = 0.025

NAMES = [
    "Ravi Menon", "Arjun Patel", "Kiran Rao", "Deepak Singh", "Neha Sharma", "Vikram Joshi",
    "Pooja Iyer", "Sanjay Gupta", "Anita Das", "Rahul Verma", "Meera Nair", "Aman Khanna",
    "Priya Kulkarni", "Rohit Bansal",
]


def build_operators(rng):
    """12 operators (one per machine per shift) + 2 managers."""
    skills = ["Beginner"] * 4 + ["Intermediate"] * 4 + ["Expert"] * 4
    rng.shuffle(skills)

    rows, assignment = [], {}
    machine_ids = list(MACHINES)
    for i in range(12):
        op_id = f"OP{i + 1:02d}"
        machine = machine_ids[i % len(machine_ids)]
        shift = "A" if i < len(machine_ids) else "B"
        assignment[(machine, shift)] = op_id
        rows.append(dict(Operator_ID=op_id, Name=NAMES[i], Role="operator", Skill_Level=skills[i]))
    for i in (12, 13):
        rows.append(dict(Operator_ID=f"OP{i + 1:02d}", Name=NAMES[i], Role="manager", Skill_Level="Expert"))

    return pd.DataFrame(rows), assignment


def build_weather(rng):
    """One weather condition per (day, shift)."""
    return {
        (day, shift): rng.choice(WEATHER, p=WEATHER_P)
        for day in range(NUM_DAYS) for shift in SHIFTS
    }


def build_telemetry(rng, assignment, skill_of):
    state = {
        m: dict(engine_hours=cfg["age"] * 1400 + rng.uniform(0, 200), hss=float(cfg["hss"]))
        for m, cfg in MACHINES.items()
    }

    rows = []
    for day in range(NUM_DAYS):
        date = START_DATE + timedelta(days=day)
        for m, service_day in SERVICE_EVENTS.items():
            if day == service_day:
                state[m]["hss"] = 0.0

        for shift, hours in SHIFTS.items():
            shift_id = f"SH-{date:%Y%m%d}-{shift}"
            for idx, hour in enumerate(hours):
                ts = date + timedelta(hours=hour)
                for m, cfg in MACHINES.items():
                    st = state[m]
                    op_id = assignment[(m, shift)]
                    mtype = cfg["type"]

                    run_h = round(READING_INTERVAL_H - rng.uniform(0, 0.2), 1)
                    st["engine_hours"] += run_h
                    st["hss"] += run_h

                    # Idle bursts cluster at the end of the shift
                    last, second_last = idx == len(hours) - 1, idx == len(hours) - 2
                    if last and rng.random() < 0.75:
                        idle = rng.normal(55, 15)
                    elif second_last and rng.random() < 0.25:
                        idle = rng.normal(35, 10)
                    else:
                        idle = rng.gamma(2.0, 6.0)
                    idle = float(np.clip(idle, 0, 110))
                    active_frac = (120 - idle) / 120

                    pressure = cfg["pressure"] - cfg["wear"] * st["hss"] + rng.normal(0, 3)
                    fuel = (
                        FUEL_RATE_LPH[mtype] * run_h * (0.35 + 0.65 * active_frac)
                        * (1 + 0.0006 * st["hss"]) * rng.normal(1, 0.05)
                    )
                    cycles = LOAD_CYCLES_PER_INTERVAL[mtype] * active_frac * rng.normal(1, 0.12)

                    zone = cfg["zone"] if rng.random() < 0.6 else rng.choice(ZONES)
                    seatbelt = rng.random() >= SEATBELT_UNFASTENED_P[skill_of[op_id]]

                    # Injected point anomalies for the Isolation Forest to find
                    if rng.random() < ANOMALY_RATE:
                        kind = rng.choice(["pressure_drop", "fuel_spike", "idle_spike"])
                        if kind == "pressure_drop":
                            pressure -= rng.uniform(35, 60)
                        elif kind == "fuel_spike":
                            fuel *= rng.uniform(1.6, 2.2)
                        else:
                            idle = float(rng.uniform(95, 115))
                            cycles *= 0.1

                    rows.append(dict(
                        Timestamp=ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        Machine_ID=m,
                        Operator_ID=op_id,
                        Shift_ID=shift_id,
                        Machine_Type=mtype,
                        GPS_Zone=zone,
                        Engine_Hours=round(st["engine_hours"], 1),
                        Fuel_Used=round(max(fuel, 0), 1),
                        Load_Cycles=int(max(round(cycles), 0)),
                        Idling_Time=int(round(idle)),
                        Hydraulic_Pressure=round(pressure, 1),
                        Hours_Since_Service=round(st["hss"], 1),
                        Seatbelt_Status=bool(seatbelt),
                    ))

    df = pd.DataFrame(rows)
    zone_counts = df.groupby(["Timestamp", "GPS_Zone"])["Machine_ID"].transform("count")
    df["Proximity_Alert"] = zone_counts > 1
    df["Safety_Alert_Triggered"] = (~df["Seatbelt_Status"]) | df["Proximity_Alert"]
    return df


def build_tasks(rng, assignment, skill_of, weather):
    # Every (day, shift, machine) slot gets MIN_TASKS_PER_SLOT tasks plus a random
    # share of the remainder, run back to back through the shift. Start times are
    # only used internally for machine age - not written out.
    slots = [(d, s, m) for d in range(NUM_DAYS) for s in SHIFTS for m in MACHINES]
    extra = rng.multinomial(NUM_TASKS - MIN_TASKS_PER_SLOT * len(slots), [1 / len(slots)] * len(slots))
    counts = extra + MIN_TASKS_PER_SLOT

    rows = []
    for (day, shift, m), count in zip(slots, counts):
        cfg = MACHINES[m]
        op_id = assignment[(m, shift)]
        skill = skill_of[op_id]
        wx = weather[(day, shift)]

        clock = START_DATE + timedelta(days=day, hours=SHIFTS[shift][0] - 1, minutes=rng.uniform(5, 20))
        for _ in range(count):
            machine_age = cfg["age"] + (clock - START_DATE).days / 365

            terrain_p = [0.20, 0.20, 0.45, 0.15] if wx == "Rainy" else [0.40, 0.25, 0.15, 0.20]
            terrain = rng.choice(TERRAIN, p=terrain_p)
            task_type = rng.choice(TASK_TYPES_BY_MACHINE[cfg["type"]])

            estimated = TASK_BASE_MINUTES[task_type] * rng.uniform(0.7, 1.4)
            estimated = int(max(5, 5 * round(estimated / 5)))

            mult = (
                1 + SKILL_EFFECT[skill] + WEATHER_EFFECT[wx] + TERRAIN_EFFECT[terrain]
                + 0.01 * machine_age + rng.normal(0, 0.05)
            )
            if wx == "Rainy" and skill == "Beginner":
                mult += RAINY_BEGINNER_EXTRA
            actual = int(round(estimated * mult))

            rows.append(dict(
                Task_ID=f"T{len(rows) + 1:04d}",
                Operator_ID=op_id,
                Machine_ID=m,
                Task_Type=task_type,
                Weather=wx,
                Site_Terrain=terrain,
                Operator_Skill=skill,
                Machine_Age=round(machine_age, 1),
                Estimated_Time=estimated,
                Actual_Time=actual,
            ))
            clock += timedelta(minutes=actual + rng.uniform(5, 15))
    return pd.DataFrame(rows)


def to_bool_strings(df, cols):
    for c in cols:
        df[c] = df[c].map({True: "true", False: "false"})
    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", default=str(Path(__file__).resolve().parent.parent))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    operators, assignment = build_operators(rng)
    skill_of = dict(zip(operators["Operator_ID"], operators["Skill_Level"]))
    weather = build_weather(rng)
    telemetry = build_telemetry(rng, assignment, skill_of)
    tasks = build_tasks(rng, assignment, skill_of, weather)

    telemetry = to_bool_strings(telemetry, ["Seatbelt_Status", "Proximity_Alert", "Safety_Alert_Triggered"])

    operators.to_csv(out / "operators.csv", index=False)
    tasks.to_csv(out / "tasks.csv", index=False)
    telemetry.to_csv(out / "telemetry.csv", index=False)

    print(f"operators.csv: {len(operators)} rows")
    print(f"tasks.csv:     {len(tasks)} rows")
    print(f"telemetry.csv: {len(telemetry)} rows")
    print(f"written to {out.resolve()}")


if __name__ == "__main__":
    main()
