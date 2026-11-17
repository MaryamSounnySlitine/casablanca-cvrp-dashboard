# Casablanca CVRP Dashboard (Streamlit)

Interactive decision dashboard for the capstone "Optimization of Urban Distribution Routes
in Casablanca: A CVRP Approach" (268 supermarkets, Tit Mellil depot, OSRM road matrix).

## How to run

    python3 -m pip install -r requirements.txt
    python3 -m streamlit run app.py

It opens at http://localhost:8501 and shows the capstone results immediately.
The route map needs an internet connection for the background map tiles.

## What you can do

- **Route map**: routes of each method on a real map; pick vehicles to highlight them.
- **Method comparison**: vehicles, distance, driving time and computation time, plus the
  Tabu Search convergence curve.
- **Vehicles**: fleet utilization gauge, load per vehicle against the 1,971 kg capacity,
  distance vs driving time per route.
- **Van planner & savings**: packs the routes into van shifts (multi-trip), with yearly
  fuel cost, savings vs Nearest Neighbor and CO2 avoided.
- **Data & export**: all route sequences and an Excel download.

Change capacity, fleet, demand (%), method and the OR-Tools time limit in the sidebar and
press **Run optimization** to re-solve. Van-planner and cost sliders update instantly.

Tabu Search uses the capstone settings (300 iterations, tenure 15, 20 neighbours, seed 4394).
