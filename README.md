# VolleyAI Coach

> A single-file volleyball coaching platform for live match tracking, player analysis, and AI-powered practice planning.

![Status](https://img.shields.io/badge/status-active-brightgreen)
![Type](https://img.shields.io/badge/type-single%20HTML%20file-orange)

---

## Overview

VolleyAI Coach is a self-contained web app — one HTML file, no install, no server. Open it in a browser and you have a full coaching platform: track live match stats, manage your roster, generate AI scouting reports, build practice plans, and review match history.

All data is stored locally in your browser (IndexedDB + localStorage). AI features use the Anthropic Claude API directly.

---

## Features

- **Live Match Tracking** — Score points, record player stats in real time, manage substitutions, and track rotations from a court view
- **Roster Management** — Add players with jersey numbers and positions (multi-position supported), view career stats per player
- **AI Player Analysis** — Generate individual scouting reports and strength/weakness breakdowns via Claude
- **Practice Plan Generator** — AI-personalised drill recommendations based on player stats and weaknesses
- **Drill Library** — Store and search your own drills by category and difficulty
- **Match History** — Review past match stats, scores, and AI-generated analysis
- **Stat Sheet Scanner** — Upload a photo of a stat sheet and AI extracts the data automatically (Claude vision)
- **Offline-first** — All data stored locally; works without internet (AI features require connection)

---

## Usage

1. Download `VolleyAI Coach.html`
2. Open it in any modern browser
3. Enter your Anthropic API key in the banner at the top to enable AI features
4. Add your roster and start coaching

No installation, no dependencies, no server required.

---

## AI Features

AI features use [Anthropic Claude](https://anthropic.com) via direct browser API call:

- **Text analysis** — Claude Haiku (player reports, practice plans, team reports)
- **Vision / OCR** — Claude Sonnet (stat sheet photo scanning)

You'll need an Anthropic API key. Your key is stored only in your browser's localStorage and never sent anywhere except directly to Anthropic.

---

## Tech Stack

| Component | Technology |
|---|---|
| App | Single HTML file (HTML + CSS + JS) |
| Charts | Chart.js (CDN) |
| Storage | IndexedDB + localStorage |
| AI | Anthropic Claude API |

---

## Project Status

VolleyAI Coach is actively developed.

- [x] Live match stat tracking
- [x] Roster & multi-position management
- [x] Court view with rotation tracking
- [x] AI player scouting reports
- [x] AI practice plan generation
- [x] Drill library
- [x] Match history & stat review
- [x] Stat sheet photo scanning (OCR)
- [x] Offline-first storage

---

*Built for coaches who want to spend less time on paperwork and more time coaching.*
