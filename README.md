---
title: Robo-Advisor Portfolio Engine
emoji: 📊
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.27.0
app_file: app.py
pinned: false
---

# Robo-Advisor Portfolio Engine

A baseline robo-advisor allocation engine covering six asset classes via
representative ETFs (VTI, VEA, VWO, BND, SCHP, VNQ), with three allocation
methods:

- **Lifecycle / Heuristic** — an age-driven ("110 minus age") glide path,
  adjusted for risk tolerance and investment horizon.
- **Mean-Variance Optimization** — a long-only optimizer personalized by
  risk tolerance and horizon via a risk-aversion-parameterized utility
  function.
- **Research-Informed** — anchored to Duarte, Fonseca, Goodman & Parker
  (2021)'s published lifecycle equity-share findings, adjusted for each
  client's own human capital (Choi, Liu & Liu, 2025) and risk tolerance.

Market data is pulled live via `yfinance` when reachable, with a
hardcoded capital-market-assumptions fallback so the app keeps working
if live data is unavailable.

Portfolio logic lives in `portfolio_engine/`, independent of the Gradio
UI in `app.py`.
