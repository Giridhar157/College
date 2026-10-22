# CineScope — Netflix Content Explorer

Intermediate class project. Flask + Pandas + Plotly + scikit-learn + SQLite + Bootstrap.
No custom JavaScript.

## Folder structure
```
cinescope/
  app.py
  requirements.txt
  cinescope.db          (auto-created on first run)
  data/netflix_titles.csv
  templates/
    base.html  index.html  browse.html  details.html
    recommendations.html  dashboard.html  watchlist.html
  static/css/style.css
```

## Setup
```bash
cd cinescope
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
```
Open http://127.0.0.1:5000

## Features
1. Explore Content — search + filters (type/genre/country/rating) + sort + pagination
2. Smart Recommendations — TF-IDF + cosine similarity on description/genre/cast/director
3. Analytics Dashboard — 5 Plotly charts
4. My Watchlist — SQLite add/remove, persistent

## Testing steps
1. Home loads, shows 8807 total.
2. Explore: search "Stranger", filter Movie + Comedies, paginate.
3. Click a title -> details + 6 similar.
4. Recommendations page -> select any title -> 10 matches with %.
5. Dashboard -> 5 charts visible.
6. Add 2 titles to watchlist, reload, remove 1 — persists after restart.
