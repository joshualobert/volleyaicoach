# VolleyAI

> AI-powered volleyball film analysis that turns your footage into actionable training plans.

![Status](https://img.shields.io/badge/status-in%20development-yellow)
![Language](https://img.shields.io/badge/language-Python-3776AB?logo=python&logoColor=white)

---

## Overview

VolleyAI is a Python script that analyses volleyball match or practice footage and generates detailed performance statistics. From those stats, it automatically builds a personalised practice plan targeting each player's specific areas for improvement.

No more guessing what to work on. Let your film tell the story.

---

## Features

- **Film Analysis** — Point the script at a video file and let the AI do the rest.
- **Performance Statistics** — Get detailed stats including attack efficiency, serve accuracy, reception quality, dig success rate, and more.
- **Tailored Practice Plans** — A personalised training programme is generated based on your performance data, targeting specific weaknesses.
- **Progress Tracking** — Run the script across multiple sessions and compare stats over time.

---

## How It Works

```
Input Video → AI Analysis → Stats Output → Practice Plan
```

1. **Provide** a match or practice video file as input.
2. **VolleyAI analyses** the footage — detecting players, tracking movements, and identifying key actions (serves, attacks, blocks, digs, sets).
3. **Stats are generated** across all key performance indicators.
4. **A custom practice plan** is printed/exported based on where the data shows the most room for improvement.

---

## Requirements

- Python 3.10+
- Dependencies listed in `requirements.txt`

---

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/volleyai.git
cd volleyai

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

```bash
python volleyai.py --input path/to/your/video.mp4
```

### Options

| Flag | Description |
|---|---|
| `--input` | Path to the video file to analyse |
| `--output` | Directory to save stats and practice plan (default: `./output`) |
| `--player` | Player name or ID to filter analysis for a specific individual |
| `--format` | Output format: `text`, `json`, or `csv` (default: `text`) |

> Options will be updated as the project develops.

---

## Output

After running, VolleyAI produces:

- **Stats report** — A breakdown of performance metrics from the analysed footage.
- **Practice plan** — A structured training plan tailored to the results.

---

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.10+ |
| Computer Vision | TBD |
| ML Framework | TBD |

---

## Project Status

VolleyAI is currently **in active development**.

- [x] Project setup
- [ ] Video input pipeline
- [ ] AI analysis model integration
- [ ] Stats engine
- [ ] Practice plan generation
- [ ] Output formatting (text / JSON / CSV)

---

*Built for players who want to train smarter, not just harder.*
