# admission_core.py
from __future__ import annotations

import io
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import pandas as pd
from sqlalchemy import Engine, text

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROGRAMS = ["PM", "IVT", "ITSS", "IB"]
SEATS = {"PM": 40, "IVT": 50, "ITSS": 30, "IB": 20}

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS applicants (
  id INTEGER PRIMARY KEY,
  phys INTEGER NOT NULL,
  rus INTEGER NOT NULL,
  math INTEGER NOT NULL,
  indiv INTEGER NOT NULL,
  total INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS loads (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  day TEXT NOT NULL,
  program TEXT NOT NULL,
  loaded_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS applications (
  day TEXT NOT NULL,
  program TEXT NOT NULL,
  applicant_id INTEGER NOT NULL,
  consent INTEGER NOT NULL,
  priority INTEGER NOT NULL,
  loaded_at TEXT NOT NULL,
  PRIMARY KEY(day, program, applicant_id),
  FOREIGN KEY(applicant_id) REFERENCES applicants(id)
);
CREATE INDEX IF NOT EXISTS idx_applications_day_program ON applications(day, program);
CREATE INDEX IF NOT EXISTS idx_applications_applicant ON applications(applicant_id);
"""


def init_db(engine: Engine) -> None:
    with engine.begin() as con:
        for stmt in SCHEMA_SQL.strip().split(";"):
            s = stmt.strip()
            if s:
                con.execute(text(s))


def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    required = ["id", "consent", "priority", "phys", "rus", "math", "indiv", "total"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")

    out = df[required].copy()
    out["id"] = out["id"].astype(int)
    out["consent"] = out["consent"].astype(int)
    out["priority"] = out["priority"].astype(int)
    for c in ["phys", "rus", "math", "indiv", "total"]:
        out[c] = out[c].astype(int)
    return out


def upsert_competition_list(engine: Engine, day: str, program: str, df: pd.DataFrame) -> None:
    """
    Implements spec 2.4:
      - delete rows absent in newest list
      - insert new applicants/applications
      - update existing rows (priority to latest upload)
    """
    df = _normalize_df(df)
    now = datetime.now().isoformat(timespec="seconds")

    with engine.begin() as con:
        con.execute(
            text("INSERT INTO loads(day, program, loaded_at) VALUES (:d,:p,:t)"),
            {"d": day, "p": program, "t": now},
        )

        # upsert applicants (scores)
        for _, r in df.iterrows():
            con.execute(
                text(
                    """
                INSERT INTO applicants(id, phys, rus, math, indiv, total)
                VALUES (:id,:ph,:ru,:ma,:in,:to)
                ON CONFLICT(id) DO UPDATE SET
                  phys=excluded.phys, rus=excluded.rus, math=excluded.math, indiv=excluded.indiv, total=excluded.total
            """
                ),
                {
                    "id": int(r.id),
                    "ph": int(r.phys),
                    "ru": int(r.rus),
                    "ma": int(r.math),
                    "in": int(r.indiv),
                    "to": int(r.total),
                },
            )

        # old ids for this (day, program)
        old_ids = set(
            x[0]
            for x in con.execute(
                text("SELECT applicant_id FROM applications WHERE day=:d AND program=:p"),
                {"d": day, "p": program},
            ).fetchall()
        )
        new_ids = set(df["id"].tolist())

        # delete absent
        to_delete = old_ids - new_ids
        if to_delete:
            con.execute(
                text(
                    f"""
                DELETE FROM applications
                WHERE day=:d AND program=:p AND applicant_id IN ({",".join(str(i) for i in to_delete)})
            """
                ),
                {"d": day, "p": program},
            )

        # upsert applications
        for _, r in df.iterrows():
            con.execute(
                text(
                    """
                INSERT INTO applications(day, program, applicant_id, consent, priority, loaded_at)
                VALUES (:d,:p,:id,:c,:pr,:t)
                ON CONFLICT(day, program, applicant_id) DO UPDATE SET
                    consent=excluded.consent,
                    priority=excluded.priority,
                    loaded_at=excluded.loaded_at
            """
                ),
                {"d": day, "p": program, "id": int(r.id), "c": int(r.consent), "pr": int(r.priority), "t": now},
            )


def query_program_list(
    engine: Engine,
    program: str,
    day: Optional[str] = None,
    consent: Optional[int] = None,
    sort: str = "total_desc",
) -> List[dict]:
    q = """
    SELECT a.applicant_id AS id, a.consent, a.priority, b.phys, b.rus, b.math, b.indiv, b.total
    FROM applications a
    JOIN applicants b ON b.id=a.applicant_id
    WHERE a.program=:p
    """
    params = {"p": program}
    if day:
        q += " AND a.day=:d"
        params["d"] = day
    if consent in (0, 1):
        q += " AND a.consent=:c"
        params["c"] = consent

    if sort == "total_asc":
        q += " ORDER BY b.total ASC, b.id ASC"
    elif sort == "id_asc":
        q += " ORDER BY b.id ASC"
    else:
        q += " ORDER BY b.total DESC, b.id ASC"

    with engine.begin() as con:
        rows = con.execute(text(q), params).mappings().all()
    return [dict(r) for r in rows]


def query_all_applicants_with_cascade(engine: Engine, day: Optional[str] = None, limit: int = 2000) -> List[dict]:
    q = """
    SELECT a.applicant_id AS id, a.program, a.priority, a.consent, b.total
    FROM applications a
    JOIN applicants b ON b.id=a.applicant_id
    """
    params = {}
    if day:
        q += " WHERE a.day=:d"
        params["d"] = day

    with engine.begin() as con:
        rows = con.execute(text(q), params).mappings().all()

    per = defaultdict(list)
    for r in rows:
        per[int(r["id"])].append(r)

    out = []
    for aid, items in per.items():
        items_sorted = sorted(items, key=lambda x: x["priority"])
        cascade = ", ".join([f'{it["program"]}:{it["priority"]}' for it in items_sorted])
        out.append(
            {
                "id": aid,
                "any_consent": any(int(it["consent"]) == 1 for it in items),
                "max_total": max(int(it["total"]) for it in items),
                "cascade": cascade,
            }
        )
    out.sort(key=lambda x: (-x["max_total"], x["id"]))
    return out[:limit]


def compute_admission(rows: List[dict]) -> Tuple[Dict[str, List[int]], Dict[str, Optional[int]]]:
    """
    rows columns: id, program, priority, consent, total
    """
    df = pd.DataFrame(rows)
    if df.empty:
        return {p: [] for p in PROGRAMS}, {p: None for p in PROGRAMS}

    df = df[df["consent"] == 1].copy()
    apps = defaultdict(list)
    scores = {}

    for _, r in df.iterrows():
        aid = int(r["id"])
        apps[aid].append((int(r["priority"]), r["program"]))
        scores[aid] = int(r["total"])

    for aid in list(apps.keys()):
        apps[aid] = [p for _, p in sorted(apps[aid], key=lambda t: t[0])]

    order = sorted(scores.keys(), key=lambda i: (-scores[i], i))
    accepted = {p: [] for p in PROGRAMS}

    for aid in order:
        for prog in apps[aid]:
            if len(accepted[prog]) < SEATS[prog]:
                accepted[prog].append(aid)
                break

    cut = {}
    for p in PROGRAMS:
        cut[p] = None if len(accepted[p]) < SEATS[p] else scores[accepted[p][-1]]
    return accepted, cut


def compute_admission_from_db(engine: Engine, day: str) -> Tuple[Dict[str, List[int]], Dict[str, Optional[int]]]:
    q = """
    SELECT a.applicant_id AS id, a.program, a.priority, a.consent, b.total
    FROM applications a
    JOIN applicants b ON b.id=a.applicant_id
    WHERE a.day=:d
    """
    with engine.begin() as con:
        rows = con.execute(text(q), {"d": day}).mappings().all()
    return compute_admission([dict(r) for r in rows])


def _stats_for_day(engine: Engine, day: str):
    # applications counts by priority (1..4) per program
    q = """
    SELECT a.program, a.priority, COUNT(*) AS cnt
    FROM applications a
    WHERE a.day=:d
    GROUP BY a.program, a.priority
    """
    with engine.begin() as con:
        appl = con.execute(text(q), {"d": day}).fetchall()
    appl_map = {(p, pr): cnt for p, pr, cnt in appl}

    accepted, cut = compute_admission_from_db(engine, day)

    # accepted by priority
    q2 = """
    SELECT a.program, a.priority, a.applicant_id, a.consent, b.total
    FROM applications a
    JOIN applicants b ON b.id=a.applicant_id
    WHERE a.day=:d AND a.consent=1
    """
    with engine.begin() as con:
        rows = con.execute(text(q2), {"d": day}).mappings().all()

    accepted_pr_counts = {(p, pr): 0 for p in PROGRAMS for pr in (1, 2, 3, 4)}

    df = pd.DataFrame(rows)
    if not df.empty:
        # normalize to compute_admission input format
        recs = []
        for r in df.to_dict("records"):
            r["id"] = int(r.get("id", r.get("applicant_id")))
            recs.append(r)

        accepted_map, _ = compute_admission(recs)
        accepted_set = {p: set(lst) for p, lst in accepted_map.items()}

        for _, r in df.iterrows():
            prog = r["program"]
            aid = int(r["applicant_id"])
            pr = int(r["priority"])
            if aid in accepted_set[prog]:
                accepted_pr_counts[(prog, pr)] += 1

    return appl_map, accepted_pr_counts, accepted, cut


def _register_cyrillic_fonts() -> Tuple[str, str]:
    """
    Registers DejaVu fonts that support Cyrillic.
    Put fonts here:
      ./fonts/DejaVuSans.ttf
      ./fonts/DejaVuSans-Bold.ttf  (желательно)
    Returns (regular_font_name, bold_font_name)
    """
    fonts_dir = Path(__file__).resolve().parent / "fonts"
    regular_path = fonts_dir / "DejaVuSans.ttf"
    bold_path = fonts_dir / "DejaVuSans-Bold.ttf"

    if not regular_path.exists():
        raise FileNotFoundError(
            "Не найден шрифт для кириллицы: fonts/DejaVuSans.ttf\n"
            "Создайте папку 'fonts' рядом с admission_core.py и положите туда DejaVuSans.ttf"
        )

    pdfmetrics.registerFont(TTFont("DejaVu", str(regular_path)))

    if bold_path.exists():
        pdfmetrics.registerFont(TTFont("DejaVu-Bold", str(bold_path)))
        return "DejaVu", "DejaVu-Bold"

    # если bold-версии нет — используем обычный
    return "DejaVu", "DejaVu"


def build_pdf_report(engine: Engine, day: str) -> bytes:
    """
    PDF with Cyrillic support (DejaVu).
    """
    # register Cyrillic fonts
    font_regular, font_bold = _register_cyrillic_fonts()

    # For dynamics: collect all days present in DB (sorted)
    with engine.begin() as con:
        days = [r[0] for r in con.execute(text("SELECT DISTINCT day FROM applications ORDER BY day")).fetchall()]

    if day not in days:
        raise ValueError("No data for requested day in DB.")

    # compute cutoffs for all days
    cutoffs_by_day = {}
    for d in days:
        _, cut = compute_admission_from_db(engine, d)
        cutoffs_by_day[d] = cut

    appl_map, accepted_pr_counts, accepted, cut = _stats_for_day(engine, day)

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    w, h = A4
    y = h - 48

    c.setFont(font_bold, 16)
    c.drawString(40, y, "Отчет по приемной кампании")
    y -= 22

    c.setFont(font_regular, 10)
    c.drawString(40, y, f"Дата/время формирования: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    y -= 16
    c.drawString(40, y, f"Срез данных: {day}")
    y -= 24

    c.setFont(font_bold, 12)
    c.drawString(40, y, "Проходные баллы")
    y -= 16

    c.setFont(font_regular, 10)
    for p in PROGRAMS:
        val = "НЕДОБОР" if cut[p] is None else str(cut[p])
        c.drawString(50, y, f"{p} (мест {SEATS[p]}): {val}")
        y -= 14
    y -= 8

    # Plot dynamics
    fig = plt.figure()
    xs = days
    for p in PROGRAMS:
        ys = [cutoffs_by_day[d][p] if cutoffs_by_day[d][p] is not None else 0 for d in xs]
        plt.plot(xs, ys, marker="o", label=p)
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Проходной балл (0 = НЕДОБОР)")
    plt.title("Динамика проходных баллов")
    plt.legend()
    plt.tight_layout()

    img_buf = io.BytesIO()
    fig.savefig(img_buf, format="png", dpi=160)
    plt.close(fig)
    img_buf.seek(0)

    from reportlab.lib.utils import ImageReader
    c.drawImage(ImageReader(img_buf), 40, y - 220, width=520, height=220, preserveAspectRatio=True, mask="auto")
    y -= 240

    # Lists of accepted
    c.setFont(font_bold, 12)
    c.drawString(40, y, "Списки зачисленных (ID и сумма баллов)")
    y -= 16

    with engine.begin() as con:
        totals = {r[0]: r[1] for r in con.execute(text("SELECT id,total FROM applicants")).fetchall()}

    for p in PROGRAMS:
        c.setFont(font_bold, 10)
        c.drawString(45, y, f"{p}:")
        y -= 12

        c.setFont(font_regular, 8)
        lst = accepted[p]

        # compact: 10-12 per line
        line = []
        for aid in lst:
            line.append(f"{aid}({totals.get(aid, '?')})")
            if len(line) == 12:
                c.drawString(55, y, "  ".join(line))
                y -= 10
                line = []
                if y < 80:
                    c.showPage()
                    y = h - 48
        if line:
            c.drawString(55, y, "  ".join(line))
            y -= 10

        y -= 6
        if y < 80:
            c.showPage()
            y = h - 48

    # Stats table
    if y < 220:
        c.showPage()
        y = h - 48

    c.setFont(font_bold, 12)
    c.drawString(40, y, "Статистика")
    y -= 16

    headers = ["Показатель"] + PROGRAMS
    rows = []
    rows.append(["Количество мест"] + [str(SEATS[p]) for p in PROGRAMS])
    rows.append(["Общее кол-во заявлений"] + [str(sum(appl_map.get((p, pr), 0) for pr in (1, 2, 3, 4))) for p in PROGRAMS])
    for pr in (1, 2, 3, 4):
        rows.append([f"Кол-во заявлений {pr}-го приоритета"] + [str(appl_map.get((p, pr), 0)) for p in PROGRAMS])
    for pr in (1, 2, 3, 4):
        rows.append([f"Кол-во зачисленных {pr}-го приоритета"] + [str(accepted_pr_counts.get((p, pr), 0)) for p in PROGRAMS])

    # draw table
    x0 = 40
    col_w = [240, 70, 70, 70, 70]
    row_h = 14

    # header row
    c.setFont(font_bold, 9)
    x = x0
    for j, hdr in enumerate(headers):
        c.rect(x, y - row_h, col_w[j], row_h, stroke=1, fill=0)
        c.drawString(x + 3, y - row_h + 3, hdr)
        x += col_w[j]
    y -= row_h

    c.setFont(font_regular, 8)
    for r in rows:
        x = x0
        for j, val in enumerate(r):
            c.rect(x, y - row_h, col_w[j], row_h, stroke=1, fill=0)
            c.drawString(x + 3, y - row_h + 3, str(val))
            x += col_w[j]
        y -= row_h
        if y < 80:
            c.showPage()
            y = h - 48

    c.showPage()
    c.save()
    return buf.getvalue()
