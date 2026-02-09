
from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
from flask import (
    Flask, render_template_string, request, send_file,
    redirect, url_for, flash
)
from sqlalchemy import create_engine, text

from admission_core import (
    PROGRAMS, SEATS,
    init_db, upsert_competition_list,
    query_program_list, query_all_applicants_with_cascade,
    compute_admission_from_db, build_pdf_report
)

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "admission.sqlite"
ENGINE = create_engine(f"sqlite:///{DB_PATH}", future=True)

app = Flask(__name__)
app.secret_key = "demo-secret-key"


TEMPLATE_INDEX = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Admission Analyzer</title>
  <link rel="stylesheet" href="{{ url_for('static', filename='styles.css') }}">
  <meta name="viewport" content="width=device-width, initial-scale=1" />
</head>
<body>
  <div class="container">
    <div class="header">
      <div class="brand">
        <div class="brand-badge"></div>
        <div>
          <h1>Admission Analyzer</h1>
          <div class="muted">Конкурсные списки • Проходные • PDF</div>
        </div>
      </div>
      <div class="badge badge-blue">Web</div>
    </div>

    <div class="grid">
      <div class="card col-6">
        <h2>Инициализация / очистка БД</h2>
        <div class="muted" style="margin-bottom:12px;">
          Создание таблиц и очистка базы для демонстрации испытаний.
        </div>
        <form method="post" action="/init" style="margin-bottom:10px;">
          <button class="btn btn-outline" type="submit">Создать таблицы (если нет)</button>
        </form>
        <form method="post" action="/clear">
          <button class="btn btn-outline" type="submit">Очистить все данные</button>
        </form>
      </div>

      <div class="card col-6">
        <h2>Загрузка CSV</h2>
        <div class="muted" style="margin-bottom:12px;">
          CSV: id, consent, priority, phys, rus, math, indiv, total
        </div>
        <form method="post" action="/upload" enctype="multipart/form-data">
          <div class="form-row">
            <div>
              <label>Дата (YYYY-MM-DD)</label>
              <input name="day" placeholder="2024-08-01" required>
            </div>
            <div>
              <label>Программа</label>
              <select name="program" required>
                {% for p in programs %}<option value="{{p}}">{{p}}</option>{% endfor %}
              </select>
            </div>
            <div>
              <label>CSV файл</label>
              <input type="file" name="file" accept=".csv" required>
            </div>
            <div>
              <button class="btn btn-primary" type="submit" style="width:100%;">Загрузить</button>
            </div>
          </div>
        </form>
      </div>

      <div class="card col-6">
        <h2>Просмотр списков</h2>
        <div class="muted" style="margin-bottom:12px;">
          Сортировка и фильтрация доступны на страницах списков.
        </div>
        <div style="display:flex; gap:10px; flex-wrap:wrap;">
          {% for p in programs %}
            <a class="btn btn-outline" href="/list/{{p}}">Список {{p}}</a>
          {% endfor %}
          <a class="btn btn-outline" href="/cascade">Единый список (каскад)</a>
        </div>
      </div>

      <div class="card col-6">
        <h2>Проходные и отчёты</h2>
        <div class="muted" style="margin-bottom:12px;">
          Рассчёт учитывает согласия и приоритеты. PDF содержит динамику и статистику.
        </div>

        <form method="get" action="/cutoffs" style="margin-bottom:10px;">
          <div class="form-row" style="grid-template-columns: 1fr auto;">
            <div>
              <label>Дата (YYYY-MM-DD)</label>
              <input name="day" placeholder="2024-08-01" required>
            </div>
            <div>
              <button class="btn btn-primary" type="submit" style="width:100%;">Рассчитать</button>
            </div>
          </div>
        </form>

        <form method="get" action="/report">
          <div class="form-row" style="grid-template-columns: 1fr auto;">
            <div>
              <label>Дата (YYYY-MM-DD)</label>
              <input name="day" placeholder="2024-08-01" required>
            </div>
            <div>
              <button class="btn btn-outline" type="submit" style="width:100%;">PDF-отчёт</button>
            </div>
          </div>
        </form>
      </div>
    </div>

    {% with messages = get_flashed_messages() %}
      {% if messages %}
        <div class="card" style="margin-top:16px;">
          <h2>Сообщения</h2>
          <ul style="margin:0; padding-left:18px;">
            {% for m in messages %}<li>{{m}}</li>{% endfor %}
          </ul>
        </div>
      {% endif %}
    {% endwith %}

    <div class="muted" style="margin-top:18px;">
      Подсказка: после запуска откройте <span class="badge badge-blue">http://127.0.0.1:5000</span>
    </div>
  </div>
</body>
</html>
"""

TEMPLATE_LIST = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>{{program}} list</title>
  <link rel="stylesheet" href="{{ url_for('static', filename='styles.css') }}">
  <meta name="viewport" content="width=device-width, initial-scale=1" />
</head>
<body>
  <div class="container">
    <div class="header">
      <div class="brand">
        <div class="brand-badge"></div>
        <div>
          <h1>Список {{program}}</h1>
          <div class="muted">{{ day or "все даты" }}</div>
        </div>
      </div>
      <a class="btn btn-outline" href="/">← назад</a>
    </div>

    <div class="card">
      <form method="get" class="form-row" style="grid-template-columns: 1fr 1fr 1fr auto;">
        <div>
          <label>Дата</label>
          <input name="day" value="{{day or ''}}" placeholder="2024-08-01">
        </div>
        <div>
          <label>Согласие</label>
          <select name="consent">
            <option value="" {% if consent is none %}selected{% endif %}>любое</option>
            <option value="1" {% if consent==1 %}selected{% endif %}>есть</option>
            <option value="0" {% if consent==0 %}selected{% endif %}>нет</option>
          </select>
        </div>
        <div>
          <label>Сортировка</label>
          <select name="sort">
            <option value="total_desc" {% if sort=="total_desc" %}selected{% endif %}>total ↓</option>
            <option value="total_asc" {% if sort=="total_asc" %}selected{% endif %}>total ↑</option>
            <option value="id_asc" {% if sort=="id_asc" %}selected{% endif %}>id ↑</option>
          </select>
        </div>
        <div>
          <button class="btn btn-primary" type="submit" style="width:100%;">Применить</button>
        </div>
      </form>
      <div class="muted" style="margin-top:10px;">
        Записей: <span class="badge badge-blue">{{rows|length}}</span>
      </div>
    </div>

    <div class="card" style="margin-top:16px;">
      <table class="table">
        <tr>
          <th>ID</th>
          <th>Согласие</th>
          <th>Приоритет</th>
          <th>Физ/ИКТ</th>
          <th>Рус</th>
          <th>Мат</th>
          <th>ИД</th>
          <th class="right">Сумма</th>
        </tr>
        {% for r in rows %}
          <tr>
            <td>{{r["id"]}}</td>
            <td>
              {% if r["consent"] %}
                <span class="badge badge-blue">да</span>
              {% else %}
                <span class="badge">нет</span>
              {% endif %}
            </td>
            <td>{{r["priority"]}}</td>
            <td class="right">{{r["phys"]}}</td>
            <td class="right">{{r["rus"]}}</td>
            <td class="right">{{r["math"]}}</td>
            <td class="right">{{r["indiv"]}}</td>
            <td class="right"><b>{{r["total"]}}</b></td>
          </tr>
        {% endfor %}
      </table>
    </div>
  </div>
</body>
</html>
"""

TEMPLATE_CASCADE = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>cascade</title>
  <link rel="stylesheet" href="{{ url_for('static', filename='styles.css') }}">
  <meta name="viewport" content="width=device-width, initial-scale=1" />
</head>
<body>
  <div class="container">
    <div class="header">
      <div class="brand">
        <div class="brand-badge"></div>
        <div>
          <h1>Единый список (каскад)</h1>
          <div class="muted">Каждый абитуриент показан один раз</div>
        </div>
      </div>
      <a class="btn btn-outline" href="/">← назад</a>
    </div>

    <div class="card">
      <form method="get" class="form-row" style="grid-template-columns: 1fr auto;">
        <div>
          <label>Дата</label>
          <input name="day" value="{{day or ''}}" placeholder="2024-08-01">
        </div>
        <div>
          <button class="btn btn-primary" type="submit" style="width:100%;">Применить</button>
        </div>
      </form>
      <div class="muted" style="margin-top:10px;">
        Показано: <span class="badge badge-blue">{{rows|length}}</span>
      </div>
    </div>

    <div class="card" style="margin-top:16px;">
      <table class="table">
        <tr>
          <th>ID</th>
          <th>Согласие (где-либо)</th>
          <th class="right">Макс. сумма</th>
          <th>Заявления (ОП:приоритет)</th>
        </tr>
        {% for r in rows %}
          <tr>
            <td>{{r["id"]}}</td>
            <td>
              {% if r["any_consent"] %}
                <span class="badge badge-blue">да</span>
              {% else %}
                <span class="badge">нет</span>
              {% endif %}
            </td>
            <td class="right"><b>{{r["max_total"]}}</b></td>
            <td class="muted">{{r["cascade"]}}</td>
          </tr>
        {% endfor %}
      </table>
    </div>
  </div>
</body>
</html>
"""

TEMPLATE_CUTOFFS = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>cutoffs</title>
  <link rel="stylesheet" href="{{ url_for('static', filename='styles.css') }}">
  <meta name="viewport" content="width=device-width, initial-scale=1" />
</head>
<body>
  <div class="container">
    <div class="header">
      <div class="brand">
        <div class="brand-badge"></div>
        <div>
          <h1>Проходные баллы</h1>
          <div class="muted">{{day}}</div>
        </div>
      </div>
      <a class="btn btn-outline" href="/">← назад</a>
    </div>

    <div class="card">
      <table class="table">
        <tr>
          <th>ОП</th>
          <th class="right">Мест</th>
          <th class="right">Проходной</th>
          <th class="right">Зачислено</th>
        </tr>
        {% for p in programs %}
          <tr>
            <td><b>{{p}}</b></td>
            <td class="right">{{seats[p]}}</td>
            <td class="right">
              {% if cutoffs[p] is none %}
                <span class="badge badge-blue">НЕДОБОР</span>
              {% else %}
                <b>{{cutoffs[p]}}</b>
              {% endif %}
            </td>
            <td class="right">{{accepted_counts[p]}}</td>
          </tr>
        {% endfor %}
      </table>

      <div class="muted" style="margin-top:12px;">
        Проходной балл — сумма баллов последнего зачисленного при заполнении мест
        (учитываются только абитуриенты с согласием; распределение — по приоритетам).
      </div>
    </div>
  </div>
</body>
</html>
"""


# -----------------------------
# Routes
# -----------------------------

@app.post("/init")
def init():
    init_db(ENGINE)
    flash("Таблицы созданы/проверены.")
    return redirect(url_for("index"))


@app.post("/clear")
def clear():
    init_db(ENGINE)
    with ENGINE.begin() as con:
        con.execute(text("DELETE FROM applications"))
        con.execute(text("DELETE FROM applicants"))
        con.execute(text("DELETE FROM loads"))
    flash("База очищена.")
    return redirect(url_for("index"))


@app.get("/")
def index():
    return render_template_string(TEMPLATE_INDEX, programs=PROGRAMS)


@app.post("/upload")
def upload():
    init_db(ENGINE)
    day = (request.form.get("day") or "").strip()
    program = (request.form.get("program") or "").strip()
    f = request.files.get("file")
    if not day or not program or not f:
        flash("Не заполнены поля.")
        return redirect(url_for("index"))

    df = pd.read_csv(f)
    upsert_competition_list(ENGINE, day, program, df)
    flash(f"Загружено: {program} на {day}.")
    return redirect(url_for("index"))


@app.get("/list/<program>")
def list_program(program: str):
    day = request.args.get("day") or None

    consent_raw = request.args.get("consent")
    consent = None
    if consent_raw in ("0", "1"):
        consent = int(consent_raw)

    sort = request.args.get("sort", "total_desc")
    rows = query_program_list(ENGINE, program, day=day, consent=consent, sort=sort)

    return render_template_string(
        TEMPLATE_LIST,
        program=program,
        day=day,
        consent=consent,
        sort=sort,
        rows=rows
    )


@app.get("/cascade")
def cascade():
    day = request.args.get("day") or None
    rows = query_all_applicants_with_cascade(ENGINE, day=day, limit=1000)
    return render_template_string(TEMPLATE_CASCADE, day=day, rows=rows)


@app.get("/cutoffs")
def cutoffs():
    day = (request.args.get("day") or "").strip()
    accepted, cut = compute_admission_from_db(ENGINE, day)
    accepted_counts = {p: len(accepted[p]) for p in PROGRAMS}

    return render_template_string(
        TEMPLATE_CUTOFFS,
        day=day,
        programs=PROGRAMS,
        seats=SEATS,
        cutoffs=cut,
        accepted_counts=accepted_counts
    )


@app.get("/report")
def report():
    day = (request.args.get("day") or "").strip()
    pdf_bytes = build_pdf_report(ENGINE, day)
    filename = f"report_{day}.pdf"
    return send_file(
        io.BytesIO(pdf_bytes),
        as_attachment=True,
        download_name=filename,
        mimetype="application/pdf"
    )


if __name__ == "__main__":
    init_db(ENGINE)
    app.run(host="127.0.0.1", port=5000, debug=True)
