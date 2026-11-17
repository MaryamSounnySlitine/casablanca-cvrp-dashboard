"""Casablanca CVRP Decision Dashboard.

Run with:  python3 -m streamlit run app.py
"""

from __future__ import annotations

import io
import math
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from solvers import nearest_neighbor, ortools_gls, tabu_search

DATA = Path(__file__).parent / "data"
DEMAND_FILE = DATA / "Casablanca_CVRP_Demand.xlsx"
MATRIX_FILE = DATA / "Casablanca_OSRM_Matrix_Cleaned_269.xlsx"
THESIS_FILE = DATA / "final_exp_output.xlsx"

NN, GLS, TABU = "Nearest Neighbor", "OR-Tools GLS", "Tabu Search"
METHODS = [GLS, TABU, NN]
METHOD_OPTIONS = ["Compare all three methods", GLS, TABU, NN]
METHOD_COLORS = {GLS: "#0E7C7B", TABU: "#E07A2E", NN: "#7A8793"}
THESIS_NAMES = {"Nearest Neighbor": NN, "Guided Local Search": GLS, "Tabu Search": TABU}

TABU_ITERATIONS, TABU_TENURE, TABU_NEIGHBOURS, TABU_SEED = 300, 15, 20, 4394
WORK_DAYS, CO2_PER_LITRE = 300, 2.68

if hasattr(go, "Scattermap"):
    MapTrace, MAP_KEY = go.Scattermap, "map"
else:
    MapTrace, MAP_KEY = go.Scattermapbox, "mapbox"

st.set_page_config(page_title="Casablanca CVRP Dashboard", page_icon="🚚", layout="wide")

st.markdown(
    """
    <style>
      .block-container { padding-top: 1.4rem; padding-bottom: 3rem; }
      .hero { background: linear-gradient(120deg, #0B4F6C 0%, #0E7C7B 55%, #3FA27F 100%);
              color: #fff; border-radius: 16px; padding: 22px 26px; margin-bottom: 18px; }
      .hero h1 { color: #fff; font-size: 1.9rem; margin: 0 0 4px 0; font-weight: 650; }
      .hero p { margin: 0; opacity: .92; font-size: .98rem; }
      .hero .tag { display:inline-block; background: rgba(255,255,255,.16); border-radius: 999px;
                   padding: 2px 10px; font-size: .78rem; margin-bottom: 8px; letter-spacing: .02em; }
      .kpis { display: grid; grid-template-columns: repeat(6, minmax(0, 1fr)); gap: 12px; margin: 6px 0 14px; }
      @media (max-width: 1100px) { .kpis { grid-template-columns: repeat(3, minmax(0, 1fr)); } }
      .kpi { background: #fff; border-radius: 14px; padding: 14px 16px; border: 1px solid #E3E8EC;
             border-top: 4px solid var(--c, #0E7C7B); }
      .kpi .l { font-size: .78rem; color: #5B6B76; text-transform: uppercase; letter-spacing: .05em; }
      .kpi .v { font-size: 1.45rem; font-weight: 650; color: #1B2A33; line-height: 1.2; }
      .kpi .s { font-size: .8rem; color: #5B6B76; }
      .pill { display:inline-block; border-radius:999px; padding:2px 10px; font-size:.8rem; font-weight:600; color:#fff; }
      div[data-testid="stTabs"] button p { font-size: 0.98rem; font-weight: 600; }
      section[data-testid="stSidebar"] { background: #EEF4F4; }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------

@st.cache_data(show_spinner="Loading data...")
def load_data():
    customers = pd.read_excel(DEMAND_FILE, sheet_name="Customers")
    depot = pd.read_excel(DEMAND_FILE, sheet_name="Depot").iloc[0]
    dist = pd.read_excel(MATRIX_FILE, sheet_name="Distance_km", index_col=0)
    dur = pd.read_excel(MATRIX_FILE, sheet_name="Duration_minutes", index_col=0)
    customers["Customer_ID"] = customers["Customer_ID"].astype(str)
    dist.index, dist.columns = dist.index.astype(str), dist.columns.astype(str)
    dur.index, dur.columns = dur.index.astype(str), dur.columns.astype(str)
    ids = ["DEPOT"] + customers["Customer_ID"].tolist()
    dist = np.rint(dist.loc[ids, ids].to_numpy(dtype=float).clip(min=0) * 1000) / 1000
    dur = dur.loc[ids, ids].to_numpy(dtype=float).clip(min=0)
    return customers.reset_index(drop=True), depot, dist, dur


@st.cache_data(show_spinner=False)
def load_thesis_routes():
    """Routes of the capstone experiment, as node indices (1..268)."""
    customers = pd.read_excel(DEMAND_FILE, sheet_name="Customers")
    index = {cid: i + 1 for i, cid in enumerate(customers["Customer_ID"].astype(str))}
    rs = pd.read_excel(THESIS_FILE, sheet_name="Route_Summary")
    res = pd.read_excel(THESIS_FILE, sheet_name="Results")
    hist = pd.read_excel(THESIS_FILE, sheet_name="Tabu_History")
    out = {}
    for method, grp in rs.groupby("Method", sort=False):
        routes = []
        for seq in grp["Route (Customer IDs)"]:
            ids = [s.strip() for s in seq.split("→")][1:-1]
            routes.append([index[c] for c in ids])
        t = res.loc[res["Method"] == method, "Execution Time (seconds)"].iloc[0]
        out[THESIS_NAMES[method]] = {"routes": routes, "time_s": float(t),
                                     "history": hist.to_dict("records") if method == "Tabu Search" else None}
    return out


customers, depot, DIST, DUR = load_data()
VCOLORS = [f"hsl({int((i * 137.508) % 360)}, 70%, 45%)" for i in range(200)]


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------

st.sidebar.markdown("### ⚙️ Model inputs")
with st.sidebar.form("inputs"):
    capacity = st.number_input("Vehicle capacity (kg)", 500, 10000, 1971, 50,
                               help="Payload of one vehicle. Default: Renault Master L2H2 (1,971 kg).")
    max_fleet = st.number_input("Available vehicles", 1, 200, 100, 1,
                                help="Maximum number of vehicles that may leave the depot.")
    demand_change = st.slider("Demand change (%)", -50, 100, 0, 5,
                              help="Applied to every store. 20 means each store orders 20% more.")
    method = st.selectbox("Solution method", METHOD_OPTIONS, index=0)
    gls_time = st.slider("OR-Tools time limit (s)", 5, 300, 60, 5,
                         help="Search time for Guided Local Search. The capstone used 120 s.")
    run = st.form_submit_button("🚀 Run optimization", type="primary", width="stretch")

st.sidebar.markdown("### 🚐 Van planner")
service_min = st.sidebar.slider("Unloading per store (min)", 0, 45, 15, 5)
shift_h = st.sidebar.slider("Shift length (h)", 4.0, 12.0, 8.0, 0.5)
reload_min = st.sidebar.slider("Reload between trips (min)", 0, 60, 20, 5)

st.sidebar.markdown("### ⛽ Cost assumptions")
fuel_l100 = st.sidebar.number_input("Fuel use (L/100 km)", 5.0, 20.0, 10.0, 0.5)
diesel_mad = st.sidebar.number_input("Diesel price (MAD/L)", 5.0, 30.0, 16.5, 0.1)

st.sidebar.caption(f"Tabu Search: {TABU_ITERATIONS} iterations, tenure {TABU_TENURE}, "
                   f"{TABU_NEIGHBOURS} neighbours, seed {TABU_SEED}. All methods minimize "
                   "vehicles first, then distance.")

demands = np.rint(customers["Demand_kg_SIMULATED"].to_numpy(dtype=float)
                  * (1 + demand_change / 100)).astype(int)
total_demand = int(demands.sum())
lower_bound = math.ceil(total_demand / capacity)


# --------------------------------------------------------------------------
# Solve
# --------------------------------------------------------------------------

def describe(method_name, routes):
    route_rows, stop_rows = [], []
    for k, r in enumerate(routes, start=1):
        full = [0] + list(r) + [0]
        km = sum(DIST[full[i], full[i + 1]] for i in range(len(full) - 1))
        mins = sum(DUR[full[i], full[i + 1]] for i in range(len(full) - 1))
        load = int(sum(demands[c - 1] for c in r))
        ids = [customers.at[c - 1, "Customer_ID"] for c in r]
        route_rows.append({"Method": method_name, "Vehicle": k, "Stores": len(r), "Load (kg)": load,
                           "Utilization (%)": round(100 * load / capacity, 1),
                           "Distance (km)": km, "Drive time (min)": mins,
                           "Route": "DEPOT → " + " → ".join(ids) + " → DEPOT"})
        for s, c in enumerate(full):
            if c == 0:
                cid, name, lat, lon, dem, cat = ("DEPOT", "Tit Mellil Depot", depot["Latitude"],
                                                 depot["Longitude"], 0, "Depot")
            else:
                row = customers.iloc[c - 1]
                cid, name, lat, lon, cat = (row["Customer_ID"], row["Shop_name"], row["Latitude"],
                                            row["Longitude"], row["Customer_Category"])
                dem = int(demands[c - 1])
            stop_rows.append({"Method": method_name, "Vehicle": k, "Stop": s, "Customer ID": cid,
                              "Store": name, "Category": cat, "Demand (kg)": dem,
                              "Latitude": lat, "Longitude": lon})
    return pd.DataFrame(route_rows), pd.DataFrame(stop_rows)


def package(results, source, inputs):
    summary_rows, rframes, sframes = [], [], []
    for m in METHODS:
        if m not in results or not results[m]["routes"]:
            continue
        rdf, sdf = describe(m, results[m]["routes"])
        rframes.append(rdf); sframes.append(sdf)
        summary_rows.append({
            "Method": m, "Vehicles": len(results[m]["routes"]),
            "Distance (km)": round(rdf["Distance (km)"].sum(), 2),
            "Drive time (h)": round(rdf["Drive time (min)"].sum() / 60, 1),
            "Avg utilization (%)": round(100 * total_demand / (len(rdf) * capacity), 1),
            "Max load (kg)": int(rdf["Load (kg)"].max()), "Stores served": int(rdf["Stores"].sum()),
            "Computation (s)": round(results[m]["time_s"], 2)})
    summary = pd.DataFrame(summary_rows)
    if NN in set(summary["Method"]) and len(summary) > 1:
        base = summary.loc[summary["Method"] == NN, "Distance (km)"].iloc[0]
        summary["Saving vs NN (%)"] = (100 * (base - summary["Distance (km)"]) / base).round(2)
    routes_all = pd.concat(rframes, ignore_index=True)
    routes_all["Distance (km)"] = routes_all["Distance (km)"].round(2)
    routes_all["Drive time (min)"] = routes_all["Drive time (min)"].round(1)
    return {"summary": summary, "routes": routes_all,
            "stops": pd.concat(sframes, ignore_index=True), "source": source, "inputs": inputs,
            "history": next((results[m]["history"] for m in results if results[m].get("history")), None)}


inputs_now = {"Vehicle capacity (kg)": capacity, "Available vehicles": max_fleet,
              "Demand change (%)": demand_change, "Total demand (kg)": total_demand,
              "Solution method": method, "OR-Tools time limit (s)": gls_time}

if "out" not in st.session_state:
    st.session_state["out"] = package(load_thesis_routes(), "capstone", inputs_now | {"Solution method": "Capstone results"})

if run:
    if demands.max() > capacity:
        st.error(f"A store needs {demands.max()} kg, more than the vehicle capacity.")
        st.stop()
    if lower_bound > max_fleet:
        st.error(f"The demand needs at least {lower_bound} vehicles but only {max_fleet} are available.")
        st.stop()
    to_run = METHODS if method == METHOD_OPTIONS[0] else [method]
    dem_list = [0] + demands.tolist()
    results = {}
    prog = st.progress(0.0, text="Optimizing routes...")
    nn = nearest_neighbor(DIST, dem_list, capacity)
    if NN in to_run:
        results[NN] = nn
    if GLS in to_run:
        prog.progress(0.2, text=f"OR-Tools Guided Local Search ({gls_time} s)...")
        results[GLS] = ortools_gls(DIST, dem_list, capacity, int(max_fleet), int(gls_time), True)
    if TABU in to_run:
        results[TABU] = tabu_search(DIST, dem_list, capacity, nn["routes"], TABU_ITERATIONS, TABU_TENURE,
                                    TABU_NEIGHBOURS, TABU_SEED,
                                    progress=lambda f: prog.progress(0.6 + 0.4 * f, text="Tabu Search..."))
    prog.empty()
    st.session_state["out"] = package(results, "run", inputs_now)

out = st.session_state["out"]
summary, routes_df, stops_df = out["summary"], out["routes"], out["stops"]
best = summary.sort_values(["Vehicles", "Distance (km)"]).iloc[0]


# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------

src_label = ("Showing the capstone results. Change the inputs and press Run optimization to re-solve."
             if out["source"] == "capstone" else "Showing your latest optimization run.")
st.markdown(
    f"""<div class="hero"><span class="tag">EGR 4394 Capstone · CVRP</span>
    <h1>🚚 Casablanca Urban Distribution Routing</h1>
    <p>Tit Mellil depot → {len(customers)} supermarkets · OSRM road distances and travel times · {src_label}</p></div>""",
    unsafe_allow_html=True)

saving = best.get("Saving vs NN (%)", float("nan"))
kpis = [
    ("Stores served", f"{int(best['Stores served'])}", "all, exactly once", "#0B4F6C"),
    ("Daily demand", f"{total_demand:,} kg", f"{demand_change:+d}% vs base" if demand_change else "simulated", "#3FA27F"),
    ("Capacity bound", f"{lower_bound}", f"vehicles at {capacity:,} kg", "#7A8793"),
    ("Best method", best["Method"].replace("OR-Tools ", ""), f"{int(best['Vehicles'])} vehicles", METHOD_COLORS[best["Method"]]),
    ("Total distance", f"{best['Distance (km)']:,.2f} km", f"{best['Drive time (h)']:.1f} h driving, all routes", "#0E7C7B"),
    ("Saving vs NN", f"{saving:.2f} %" if saving == saving else "—", "distance per day", "#E07A2E"),
]
st.markdown('<div class="kpis">' + "".join(
    f'<div class="kpi" style="--c:{c}"><div class="l">{l}</div><div class="v">{v}</div><div class="s">{s}</div></div>'
    for l, v, s, c in kpis) + "</div>", unsafe_allow_html=True)

tab_map, tab_cmp, tab_veh, tab_van, tab_data = st.tabs(
    ["🗺️ Route map", "📊 Method comparison", "🚛 Vehicles", "🕒 Van planner & savings", "📥 Data & export"])


def plotly_base(fig, h=360):
    fig.update_layout(height=h, margin=dict(l=10, r=10, t=30, b=10), plot_bgcolor="#FFFFFF",
                      paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#1B2A33"),
                      legend=dict(orientation="h", y=1.08, x=0))
    fig.update_xaxes(gridcolor="#E9EEF1"); fig.update_yaxes(gridcolor="#E9EEF1")
    return fig


# --------------------------------------------------------------------------
# Map tab
# --------------------------------------------------------------------------

with tab_map:
    c1, c2 = st.columns([1, 2])
    shown = c1.radio("Method", summary["Method"].tolist(), horizontal=True,
                     index=summary["Method"].tolist().index(best["Method"]))
    rsub = routes_df[routes_df["Method"] == shown]
    ssub = stops_df[stops_df["Method"] == shown]
    vehicles = c2.multiselect("Vehicles to show (empty = all)", rsub["Vehicle"].tolist(),
                              format_func=lambda v: f"Vehicle {v}")
    show_legs = st.checkbox("Show the drives to and from the depot", value=True,
                            help="Thin, faint lines from the Tit Mellil depot to the first store and back from the last store.")
    left, right = st.columns([3, 2])
    with left:
        fig = go.Figure()
        for v, grp in ssub.groupby("Vehicle"):
            grp = grp.sort_values("Stop")
            on = not vehicles or v in vehicles
            picked = bool(vehicles) and v in vehicles
            info = rsub[rsub["Vehicle"] == v].iloc[0]
            body = grp.iloc[1:-1]
            if show_legs:
                # drives from the depot to the first store and back from the last store: thin and faint
                leg_lat = [grp["Latitude"].iloc[0], grp["Latitude"].iloc[1], None,
                           grp["Latitude"].iloc[-2], grp["Latitude"].iloc[-1]]
                leg_lon = [grp["Longitude"].iloc[0], grp["Longitude"].iloc[1], None,
                           grp["Longitude"].iloc[-2], grp["Longitude"].iloc[-1]]
                fig.add_trace(MapTrace(lat=leg_lat, lon=leg_lon, mode="lines",
                                       line=dict(width=2 if picked else 1, color=VCOLORS[v - 1]),
                                       opacity=0.85 if picked else (0.18 if on else 0.04),
                                       hoverinfo="skip", showlegend=False))
            hover = [f"<b>{name}</b><br>{cid} · {cat}<br>Vehicle {v}, stop {s} · {d} kg"
                     for cid, name, cat, s, d in zip(body["Customer ID"], body["Store"], body["Category"],
                                                     body["Stop"], body["Demand (kg)"])]
            fig.add_trace(MapTrace(
                lat=body["Latitude"], lon=body["Longitude"], mode="lines+markers",
                line=dict(width=4 if picked else 2.5, color=VCOLORS[v - 1]),
                marker=dict(size=[6 + d / 90 for d in body["Demand (kg)"]], color=VCOLORS[v - 1]),
                opacity=1 if on else 0.12, name=f"V{v}: {info['Stores']} stores, {info['Distance (km)']} km",
                text=hover, hoverinfo="text"))
        fig.add_trace(MapTrace(lat=[depot["Latitude"]], lon=[depot["Longitude"]], mode="markers+text",
                               marker=dict(size=18, color="#111111"), text=["Depot"], textposition="top right",
                               name="Tit Mellil Depot", hoverinfo="name"))
        lat_mid = float((ssub["Latitude"].min() + ssub["Latitude"].max()) / 2)
        lon_mid = float((ssub["Longitude"].min() + ssub["Longitude"].max()) / 2)
        fig.update_layout(**{MAP_KEY: dict(style="carto-positron", zoom=10.7,
                                           center=dict(lat=lat_mid, lon=lon_mid))},
                          margin=dict(l=0, r=0, t=0, b=0), height=560, showlegend=False)
        st.plotly_chart(fig, width="stretch", config={"displaylogo": False, "scrollZoom": True})
        st.caption("Each colour is one vehicle's delivery loop in the city. Faint lines are the drives to and from the Tit Mellil depot. Marker size grows with store demand. Scroll to zoom, hover for details.")
    with right:
        view = rsub if not vehicles else rsub[rsub["Vehicle"].isin(vehicles)]
        st.dataframe(view[["Vehicle", "Stores", "Load (kg)", "Utilization (%)", "Distance (km)", "Drive time (min)"]],
                     hide_index=True, width="stretch", height=380,
                     column_config={"Utilization (%)": st.column_config.ProgressColumn(
                         "Utilization", min_value=0, max_value=100, format="%.1f%%")})
        for v in vehicles[:3]:
            seq = ssub[(ssub["Vehicle"] == v)].sort_values("Stop")["Store"].tolist()
            st.markdown(f"**Vehicle {v}:** " + " → ".join(seq))


# --------------------------------------------------------------------------
# Comparison tab
# --------------------------------------------------------------------------

with tab_cmp:
    if len(summary) < 2:
        st.info("Run 'Compare all three methods' to see the comparison.")
    order = [m for m in METHODS if m in set(summary["Method"])]
    srt = summary.set_index("Method").loc[order].reset_index()
    cols = st.columns(2)
    specs = [("Vehicles", "Vehicles used", lower_bound, "Capacity bound"),
             ("Distance (km)", "Total distance (km per day)", None, None),
             ("Drive time (h)", "Driving time of all routes added together (h per day)", None, None),
             ("Computation (s)", "Computation time (s)", None, None)]
    for i, (col, title, ref, ref_label) in enumerate(specs):
        f = go.Figure(go.Bar(x=srt[col], y=srt["Method"], orientation="h",
                             marker_color=[METHOD_COLORS[m] for m in srt["Method"]],
                             text=[f"{v:,.2f}" if col != "Drive time (h)" else f"{v:,.1f}" for v in srt[col]] if col != "Vehicles" else [str(int(v)) for v in srt[col]],
                             textposition="outside", cliponaxis=False))
        if ref is not None:
            f.add_vline(x=ref, line_dash="dash", line_color="#1B2A33", annotation_text=ref_label, annotation_position="bottom left")
        if col == "Vehicles":
            f.update_xaxes(range=[min(lower_bound, srt[col].min()) - 4, srt[col].max() + 2])
        elif col in ("Distance (km)", "Drive time (h)"):
            f.update_xaxes(range=[srt[col].min() * 0.9, srt[col].max() * 1.06])
        else:
            f.update_xaxes(range=[0, srt[col].max() * 1.2])
        f.update_layout(title=title, yaxis=dict(autorange="reversed"))
        cols[i % 2].plotly_chart(plotly_base(f, 260), width="stretch", config={"displaylogo": False})
    st.dataframe(srt, hide_index=True, width="stretch")
    if out.get("history"):
        h = pd.DataFrame(out["history"])
        f = go.Figure(go.Scatter(x=h["Iteration"], y=h["Distance_km"], mode="lines",
                                 line=dict(color=METHOD_COLORS[TABU], width=3), name="Tabu Search best"))
        for m in (GLS, NN):
            if m in set(srt["Method"]):
                f.add_hline(y=float(srt.loc[srt["Method"] == m, "Distance (km)"].iloc[0]), line_dash="dash",
                            line_color=METHOD_COLORS[m], annotation_text=m)
        f.update_layout(title="Tabu Search convergence (best distance by iteration)",
                        xaxis_title="Iteration", yaxis_title="km")
        st.plotly_chart(plotly_base(f, 320), width="stretch", config={"displaylogo": False})


# --------------------------------------------------------------------------
# Vehicles tab
# --------------------------------------------------------------------------

with tab_veh:
    shown_v = st.radio("Method ", summary["Method"].tolist(), horizontal=True,
                       index=summary["Method"].tolist().index(best["Method"]), key="veh_method")
    rv = routes_df[routes_df["Method"] == shown_v].sort_values("Load (kg)", ascending=False)
    g1, g2 = st.columns([1, 2])
    util = 100 * rv["Load (kg)"].sum() / (len(rv) * capacity)
    gauge = go.Figure(go.Indicator(mode="gauge+number", value=util, number={"suffix": " %"},
                                   title={"text": "Average fleet utilization"},
                                   gauge={"axis": {"range": [0, 100]}, "bar": {"color": METHOD_COLORS[shown_v]},
                                          "steps": [{"range": [0, 70], "color": "#F3E1E1"},
                                                    {"range": [70, 90], "color": "#F6EDD9"},
                                                    {"range": [90, 100], "color": "#DDEFE6"}]}))
    g1.plotly_chart(plotly_base(gauge, 300), width="stretch", config={"displaylogo": False})
    f = go.Figure(go.Bar(x=[f"V{v}" for v in rv["Vehicle"]], y=rv["Load (kg)"],
                         marker_color=[VCOLORS[v - 1] for v in rv["Vehicle"]],
                         hovertext=[f"Vehicle {v}: {l:,} kg, {s} stores, {d} km" for v, l, s, d in
                                    zip(rv["Vehicle"], rv["Load (kg)"], rv["Stores"], rv["Distance (km)"])],
                         hoverinfo="text"))
    f.add_hline(y=capacity, line_dash="dash", line_color="#1B2A33", annotation_text=f"Capacity {capacity:,} kg")
    f.update_layout(title="Load per vehicle (sorted)", yaxis_title="kg")
    g2.plotly_chart(plotly_base(f, 300), width="stretch", config={"displaylogo": False})
    sc = go.Figure(go.Scatter(x=rv["Distance (km)"], y=rv["Drive time (min)"], mode="markers",
                              marker=dict(size=rv["Stores"] * 3, color=[VCOLORS[v - 1] for v in rv["Vehicle"]],
                                          line=dict(width=1, color="#fff")),
                              text=[f"Vehicle {v}: {s} stores" for v, s in zip(rv["Vehicle"], rv["Stores"])],
                              hoverinfo="text+x+y"))
    sc.update_layout(title="Route distance vs driving time (bubble size = stores)",
                     xaxis_title="km", yaxis_title="minutes")
    st.plotly_chart(plotly_base(sc, 320), width="stretch", config={"displaylogo": False})


# --------------------------------------------------------------------------
# Van planner tab
# --------------------------------------------------------------------------

def pack_vans(rdf):
    trips = sorted(({"Vehicle": r["Vehicle"], "Stores": r["Stores"],
                     "minutes": r["Drive time (min)"] + service_min * r["Stores"]} for _, r in rdf.iterrows()),
                   key=lambda t: -t["minutes"])
    vans, shift = [], shift_h * 60
    for t in trips:
        van = next((v for v in vans if v["used"] + reload_min + t["minutes"] <= shift), None)
        if van is None:
            van = {"used": -reload_min, "trips": []}
            vans.append(van)
        start = van["used"] + reload_min
        van["trips"].append(t | {"start": start})
        van["used"] = start + t["minutes"]
    return vans


with tab_van:
    st.markdown("Each route is a short loop (about one hour of driving), so **one van can drive several routes per shift**. "
                "The planner packs the routes into van shifts, longest first. Adjust unloading time and shift length in the sidebar.")
    vm = st.radio("Method  ", summary["Method"].tolist(), horizontal=True,
                  index=summary["Method"].tolist().index(best["Method"]), key="van_method")
    vans = pack_vans(routes_df[routes_df["Method"] == vm])
    n_routes = sum(len(v["trips"]) for v in vans)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Routes to drive", n_routes)
    k2.metric("Vans needed", len(vans), delta=f"{len(vans) - n_routes} vs one van per route", delta_color="inverse")
    k3.metric("Trips per van", f"{n_routes / len(vans):.1f}")
    k4.metric("Longest route", f"{max(t['minutes'] for v in vans for t in v['trips']) / 60:.2f} h")
    gf = go.Figure()
    for i, v in enumerate(vans, start=1):
        for t in v["trips"]:
            gf.add_trace(go.Bar(y=[f"Van {i}"], x=[t["minutes"] / 60], base=[t["start"] / 60], orientation="h",
                                marker_color=VCOLORS[t["Vehicle"] - 1], text=f"R{t['Vehicle']}",
                                textposition="inside", textangle=0, insidetextanchor="start", hovertext=f"Route {t['Vehicle']}: {t['Stores']} stores, "
                                f"{t['start'] / 60:.1f} h → {(t['start'] + t['minutes']) / 60:.1f} h",
                                hoverinfo="text", showlegend=False))
    gf.add_vline(x=shift_h, line_dash="dash", line_color="#1B2A33", annotation_text="End of shift")
    gf.update_layout(barmode="overlay", title="Van schedules", xaxis_title="hours into the shift",
                     yaxis=dict(autorange="reversed"))
    st.plotly_chart(plotly_base(gf, max(300, 26 * len(vans) + 80)), width="stretch",
                    config={"displaylogo": False})

    st.markdown("#### ⛽ Fuel, cost and CO₂ per year")
    fuel = summary[["Method", "Distance (km)"]].copy()
    fuel["Fuel cost / year (MAD)"] = (fuel["Distance (km)"] * WORK_DAYS * fuel_l100 / 100 * diesel_mad).round(0)
    if NN in set(fuel["Method"]):
        nn_km = float(fuel.loc[fuel["Method"] == NN, "Distance (km)"].iloc[0])
        litres = (nn_km - fuel["Distance (km)"]) * WORK_DAYS * fuel_l100 / 100
        fuel["Diesel saved vs NN (L/yr)"] = litres.round(0)
        fuel["Saved vs NN (MAD/yr)"] = (litres * diesel_mad).round(0)
        fuel["CO₂ avoided (t/yr)"] = (litres * CO2_PER_LITRE / 1000).round(1)
    st.dataframe(fuel, hide_index=True, width="stretch")
    st.caption(f"{WORK_DAYS} working days, {CO2_PER_LITRE} kg CO₂ per litre of diesel. Estimates for discussion.")


# --------------------------------------------------------------------------
# Data tab
# --------------------------------------------------------------------------

with tab_data:
    st.markdown("#### Inputs used")
    st.dataframe(pd.DataFrame({"Input": list(out["inputs"].keys()),
                               "Value": [str(v) for v in out["inputs"].values()]}),
                 hide_index=True, width="stretch")
    st.markdown("#### Full route sequences")
    st.dataframe(routes_df[["Method", "Vehicle", "Stores", "Load (kg)", "Distance (km)", "Route"]],
                 hide_index=True, width="stretch", height=320)

    def to_excel():
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as xw:
            pd.DataFrame({"Input": list(out["inputs"].keys()),
                          "Value": [str(v) for v in out["inputs"].values()]}).to_excel(xw, sheet_name="Inputs", index=False)
            summary.to_excel(xw, sheet_name="Summary", index=False)
            routes_df.to_excel(xw, sheet_name="Routes", index=False)
            stops_df.to_excel(xw, sheet_name="Stops", index=False)
        return buf.getvalue()

    st.download_button("📥 Download results (Excel)", to_excel(), file_name="cvrp_results.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                       type="primary")
