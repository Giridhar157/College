"""
CineScope — Netflix Content Explorer
Beginner-friendly Flask app. No custom JavaScript.
Features: Browse/Search, Details, Smart Recommendations (TF-IDF),
Analytics Dashboard (Plotly), My Watchlist (SQLite).
Run: python app.py  ->  http://127.0.0.1:5000
"""
import os
import sqlite3
from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.io as pio
from flask import Flask, g, redirect, render_template, request, url_for
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "netflix_titles.csv")


def _pick_db_path():
    """Pick a writable SQLite path. Falls back to /tmp on locked/network filesystems."""
    candidates = [os.path.join(BASE_DIR, "cinescope.db"), "/tmp/cinescope.db"]
    for p in candidates:
        try:
            con = sqlite3.connect(p, timeout=5)
            con.execute("CREATE TABLE IF NOT EXISTS __wtest__(a TEXT)")
            con.execute("DROP TABLE __wtest__")
            con.commit()
            con.close()
            return p
        except Exception:
            continue
    return "/tmp/cinescope.db"


DB_PATH = _pick_db_path()

app = Flask(__name__)

# -------------------------------------------------
# 1. Load & clean data (once at startup)
# -------------------------------------------------
df = pd.read_csv(DATA_PATH, dtype=str, keep_default_na=False)
# release_year as int where possible
df["release_year"] = pd.to_numeric(df["release_year"], errors="coerce")
df["title_search"] = df["title"].str.lower()
df["director"] = df["director"].fillna("")
df["cast"] = df["cast"].fillna("")
df["country"] = df["country"].fillna("")
df["rating"] = df["rating"].replace("", "Unrated")
df["listed_in"] = df["listed_in"].replace("", "Unknown")
df["description"] = df["description"].replace("", "No description available.")

# Combined text for recommendations
df["combined"] = (
    df["description"] + " "
    + df["listed_in"] + " "
    + df["cast"] + " "
    + df["director"] + " "
    + df["type"]
)

# TF-IDF model (built once)
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
tfidf_matrix = vectorizer.fit_transform(df["combined"])
# fast lookup: show_id -> row index
id_to_index = {sid: i for i, sid in enumerate(df["show_id"])}

# Filter dropdown values
ALL_GENRES = sorted({g.strip() for s in df["listed_in"] for g in s.split(",") if g.strip()})
country_counts = df[df["country"] != ""].copy()
country_counts["country_split"] = country_counts["country"].str.split(",")
all_countries = []
for val in country_counts["country_split"]:
    for c in val:
        c = c.strip()
        if c:
            all_countries.append(c)
TOP_COUNTRIES = pd.Series(all_countries).value_counts().head(30).index.tolist()
ALL_RATINGS = sorted(df["rating"].unique().tolist())
MIN_YEAR = int(df["release_year"].min())
MAX_YEAR = int(df["release_year"].max())

# -------------------------------------------------
# 2. SQLite watchlist helpers
# -------------------------------------------------
def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
    return g.db

@app.teardown_appcontext
def close_db(_exc=None):
    db = g.pop("db", None)
    if db is not None:
        db.close()

def init_db():
    con = sqlite3.connect(DB_PATH)
    con.execute("""
        CREATE TABLE IF NOT EXISTS watchlist (
            show_id TEXT PRIMARY KEY,
            title TEXT,
            type TEXT,
            added_on TEXT
        )
    """)
    con.commit()
    con.close()

init_db()

def watchlist_ids():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    rows = con.execute("SELECT show_id FROM watchlist").fetchall()
    con.close()
    return {r["show_id"] for r in rows}

# -------------------------------------------------
# 3. Recommendation logic
# -------------------------------------------------
def get_recommendations(show_id, n=10):
    """Return n most similar titles using cosine similarity on TF-IDF."""
    if show_id not in id_to_index:
        return []
    idx = id_to_index[show_id]
    scores = linear_kernel(tfidf_matrix[idx], tfidf_matrix).flatten()
    # top n+1 then drop itself
    top_idx = scores.argsort()[::-1][1:n + 1]
    recs = df.iloc[top_idx].copy()
    recs["score"] = scores[top_idx]
    return recs.to_dict(orient="records")

# -------------------------------------------------
# 4. Routes
# -------------------------------------------------
@app.route("/")
def home():
    total = len(df)
    movies = int((df["type"] == "Movie").sum())
    shows = int((df["type"] == "TV Show").sum())
    countries = len(set(all_countries))
    genres = len(ALL_GENRES)
    # Featured: 6 newest titles
    featured = df.sort_values("release_year", ascending=False).head(6).to_dict(orient="records")
    wl = watchlist_ids()
    return render_template("index.html", total=total, movies=movies, shows=shows,
                           countries=countries, genres=genres,
                           featured=featured, wl=wl)

@app.route("/browse")
def browse():
    q = request.args.get("q", "").strip()
    f_type = request.args.get("type", "All")
    f_genre = request.args.get("genre", "All")

    f_country = request.args.get("country", "All")
    f_rating = request.args.get("rating", "All")
    sort = request.args.get("sort", "Newest")
    page = max(1, int(request.args.get("page", 1) or 1))
    PER_PAGE = 12

    out = df
    if q:
        out = out[out["title"].str.lower().str.contains(q.lower(), na=False)
                  | out["cast"].str.lower().str.contains(q.lower(), na=False)
                  | out["director"].str.lower().str.contains(q.lower(), na=False)
                  | out["description"].str.lower().str.contains(q.lower(), na=False)]
    if f_type != "All":
        out = out[out["type"] == f_type]
    if f_genre != "All":
        out = out[out["listed_in"].str.contains(f_genre, na=False)]
    if f_country != "All":
        out = out[out["country"].str.contains(f_country, na=False)]
    if f_rating != "All":
        out = out[out["rating"] == f_rating]

    if sort == "Newest":
        out = out.sort_values(["release_year", "title"], ascending=[False, True])
    elif sort == "Oldest":
        out = out.sort_values(["release_year", "title"], ascending=[True, True])
    elif sort == "A-Z":
        out = out.sort_values("title")

    total = len(out)
    pages = max(1, (total + PER_PAGE - 1) // PER_PAGE)
    page = min(page, pages)
    start, end = (page - 1) * PER_PAGE, page * PER_PAGE
    results = out.iloc[start:end].to_dict(orient="records")
    wl = watchlist_ids()
    return render_template("browse.html", results=results, total=total,
                           page=page, pages=pages, q=q,
                           f_type=f_type, f_genre=f_genre, f_country=f_country,
                           f_rating=f_rating, sort=sort,
                           genres=ALL_GENRES, countries=TOP_COUNTRIES,
                           ratings=ALL_RATINGS, wl=wl)

@app.route("/title/<show_id>")
def details(show_id):
    if show_id not in id_to_index:
        return "Title not found", 404
    row = df.iloc[id_to_index[show_id]].to_dict()
    recs = get_recommendations(show_id, n=6)
    wl = watchlist_ids()
    in_list = show_id in wl
    return render_template("details.html", t=row, recs=recs, in_list=in_list)

@app.route("/recommendations")
def recommendations():
    show_id = request.args.get("show_id", "").strip()
    selected = None
    recs = []
    # list for dropdown: show_id - title
    all_titles = df[["show_id", "title", "release_year", "type"]].sort_values("title").to_dict(orient="records")
    if show_id and show_id in id_to_index:
        selected = df.iloc[id_to_index[show_id]].to_dict()
        recs = get_recommendations(show_id, n=10)
    wl = watchlist_ids()
    return render_template("recommendations.html", all_titles=all_titles,
                           selected=selected, recs=recs, show_id=show_id, wl=wl)

@app.route("/dashboard")
def dashboard():
    pio.templates.default = "plotly_dark"
    # 1. Movies vs Shows
    fig1 = px.pie(df, names="type", title="Movies vs TV Shows",
                  color_discrete_sequence=["#E50914", "#564d4d"])
    # 2. Top 10 genres
    g = pd.Series([x.strip() for s in df["listed_in"] for x in s.split(",")])
    g = g.value_counts().head(10).reset_index()
    g.columns = ["Genre", "Count"]
    fig2 = px.bar(g, x="Count", y="Genre", orientation="h", title="Top 10 Genres",
                  color="Count", color_continuous_scale="Reds")
    fig2.update_layout(yaxis={"categoryorder": "total ascending"})
    # 3. Top 10 countries
    c = pd.Series(all_countries).value_counts().head(10).reset_index()
    c.columns = ["Country", "Count"]
    fig3 = px.bar(c, x="Country", y="Count", title="Top 10 Producing Countries",
                  color="Count", color_continuous_scale="Reds")
    # 4. Ratings
    r = df["rating"].value_counts().head(10).reset_index()
    r.columns = ["Rating", "Count"]
    fig4 = px.bar(r, x="Rating", y="Count", title="Content by Rating",
                  color="Count", color_continuous_scale="Reds")
    # 5. Releases over time
    y = df["release_year"].value_counts().sort_index().reset_index()
    y.columns = ["Year", "Count"]
    y = y[y["Year"] >= 2000]
    fig5 = px.line(y, x="Year", y="Count", title="Releases Over Time (2000+)", markers=True)
    fig5.update_traces(line_color="#E50914")

    charts = [fig1, fig2, fig3, fig4, fig5]
    html_charts = [pio.to_html(f, full_html=False, include_plotlyjs="cdn") for f in charts]
    stats = {
        "total": len(df),
        "movies": int((df["type"] == "Movie").sum()),
        "shows": int((df["type"] == "TV Show").sum()),
        "countries": len(set(all_countries)),
        "oldest": MIN_YEAR,
        "newest": MAX_YEAR,
    }
    return render_template("dashboard.html", charts=html_charts, stats=stats)

@app.route("/watchlist")
def watchlist():
    con = get_db()
    rows = con.execute("SELECT * FROM watchlist ORDER BY added_on DESC").fetchall()
    items = []
    for r in rows:
        d = dict(r)
        if d["show_id"] in id_to_index:
            full = df.iloc[id_to_index[d["show_id"]]].to_dict()
            d.update(full)
        items.append(d)
    return render_template("watchlist.html", items=items)

@app.route("/watchlist/add/<show_id>", methods=["POST"])
def watchlist_add(show_id):
    if show_id in id_to_index:
        row = df.iloc[id_to_index[show_id]]
        con = get_db()
        con.execute("INSERT OR IGNORE INTO watchlist(show_id,title,type,added_on) VALUES(?,?,?,?)",
                    (show_id, row["title"], row["type"], datetime.now().strftime("%Y-%m-%d %H:%M")))
        con.commit()
    return redirect(request.referrer or url_for("watchlist"))

@app.route("/watchlist/remove/<show_id>", methods=["POST"])
def watchlist_remove(show_id):
    con = get_db()
    con.execute("DELETE FROM watchlist WHERE show_id=?", (show_id,))
    con.commit()
    return redirect(request.referrer or url_for("watchlist"))


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)