# -*- coding: utf-8 -*-
"""
«Есте сақтау жолы» — ИИ-генератор маршрутов по местам памяти Казахстана.

Как работает приложение (кратко для защиты):
  1. Браузер открывает страницу (GET /) и показывает форму.
  2. Пользователь заполняет форму -> JS отправляет POST /api/route.
  3. Сервер читает places.json, оставляет объекты по интересам пользователя
     и отправляет их вместе с параметрами в OpenAI (gpt-4o-mini).
  4. ИИ возвращает маршрут в формате JSON. Сервер ПРОВЕРЯЕТ ответ:
     неизвестные id (выдуманные ИИ объекты) отбрасываются.
  5. Сервер добавляет к каждой остановке данные из places.json
     (название, координаты, описание) и отдаёт результат браузеру.

Ключ OPENAI_API_KEY хранится только на сервере (файл .env) и во фронтенд не попадает.
"""

import json
import os

from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_from_directory
from openai import (
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    OpenAI,
    OpenAIError,
    RateLimitError,
)

# ---------------------------------------------------------------------------
# Настройка
# ---------------------------------------------------------------------------

# Загружаем переменные из файла .env (там лежит OPENAI_API_KEY)
load_dotenv()

# Папка, где лежит app.py — от неё строим пути к файлам
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PLACES_PATH = os.path.join(BASE_DIR, "places.json")
CITIES_PATH = os.path.join(BASE_DIR, "cities.json")

app = Flask(__name__)
# Чтобы казахские буквы в JSON не превращались в \uXXXX
app.json.ensure_ascii = False

MODEL = "gpt-4o-mini"

# Допустимые значения полей формы (серверная валидация — на фронтенд полагаться нельзя)
# Список пунктов отправления лежит в cities.json: города, аудандар и ауылдар ҚЗО
ALLOWED_AGE_GROUPS = ["оқушы", "ересек", "шетелдік турист"]
# Интерес пользователя -> категория объектов в places.json
ALLOWED_INTERESTS = {
    "repressions": "саяси қуғын-сүргін",
    "nuclear": "ядролық сынақтар",
    "famine": "1930-жылдардағы ашаршылық",
    "war": "Ұлы Отан соғысы",
    "ecology": "экологиялық апаттар",
}

# Системный промпт для GPT (вставлен как есть)
SYSTEM_PROMPT = """Сен — Қазақстанның тарихи жады орындары бойынша сыпайы әрі білікті экскурсоводсың. Тек берілген places.json тізіміндегі нысандарды қолдан, жаңа нысан ойлап таппа. Пайдаланушының қаласын, күн санын, жас тобын және қызығушылығын ескеріп, логикалық маршрут құр (жақын нысандарды бір күнге топта). Оқушыларға қарапайым тілмен, шетелдіктерге контекстпен түсіндір. Әр нысанға 2-3 мінез-құлық ережесін жаз (мысалы: тыныштық сақтау, гүл қою, рұқсатсыз фото түсірмеу). Трагедияны ойын-сауық ретінде ұсынба. Жауапты тек қазақ тілінде, мына JSON форматында бер:
{ "title": "...", "days": [ { "day": 1, "stops": [ { "id": "...", "time": "10:00–12:00", "note": "...", "rules": ["...", "..."] } ] } ], "why_important": "..." }"""

# Сообщения об ошибках на казахском языке
ERR_NO_KEY = "Сервер бапталмаған: .env файлында OPENAI_API_KEY табылмады."
ERR_AUTH = "OpenAI кілті жарамсыз. .env файлындағы кілтті тексеріңіз."
ERR_RATE = "OpenAI сұраныс шегі немесе шоты таусылды. Біраздан кейін қайталап көріңіз."
ERR_NETWORK = "OpenAI қызметімен байланыс орнатылмады. Интернет байланысын тексеріңіз."
ERR_TIMEOUT = "ИИ жауап беруге ұзақ уақыт алды. Қайтадан көріңіз."
ERR_OTHER = "ИИ қызметінде қате болды. Кейінірек қайталап көріңіз."
ERR_BAD_ANSWER = "ИИ жауабы дұрыс форматта келмеді. «Маршрут құру» батырмасын қайта басыңыз."
ERR_EMPTY = "Таңдалған қызығушылық бойынша маршрут құру мүмкін болмады. Басқа параметрлерді көріңіз."


def load_places():
    """Читает places.json и возвращает список объектов."""
    with open(PLACES_PATH, encoding="utf-8") as f:
        return json.load(f)


def load_cities():
    """Читает cities.json: пункты отправления (города, аудандар и ауылдар ҚЗО)."""
    with open(CITIES_PATH, encoding="utf-8") as f:
        return json.load(f)


def error(message, status):
    """Единый формат ответа с ошибкой: {"error": "..."} + HTTP-код."""
    return jsonify({"error": message}), status


# ---------------------------------------------------------------------------
# Маршруты Flask
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    """GET / — отдаём страницу index.html."""
    return send_from_directory(BASE_DIR, "index.html")


@app.route("/api/cities")
def get_cities():
    """GET /api/cities — список пунктов отправления для поиска в форме."""
    return jsonify(load_cities())


@app.route("/api/route", methods=["POST"])
def build_route():
    """POST /api/route — принимает параметры формы и возвращает маршрут."""

    # --- 1. Читаем и проверяем параметры формы -----------------------------
    data = request.get_json(silent=True) or {}
    city_name = data.get("city")
    age_group = data.get("age_group")
    interests = data.get("interests") or []

    try:
        days = int(data.get("days"))
    except (TypeError, ValueError):
        days = 0

    # Пункт отправления должен быть в cities.json
    city_info = next((c for c in load_cities() if c["name"] == city_name), None)
    if city_info is None:
        return error("Шығатын қала дұрыс таңдалмаған. Тізімден таңдаңыз.", 400)
    # Для ИИ пишем пункт вместе с областью: «Пырымов ауылы (Күмжиек), Қызылорда облысы, Қазалы ауданы»
    city = f"{city_info['name']} ({city_info['region']})"
    if not 1 <= days <= 5:
        return error("Күн саны 1 мен 5 аралығында болуы керек.", 400)
    if age_group not in ALLOWED_AGE_GROUPS:
        return error("Жас тобы дұрыс таңдалмаған.", 400)
    # Оставляем только известные интересы
    interests = [i for i in interests if i in ALLOWED_INTERESTS]
    if not interests:
        return error("Кемінде бір қызығушылықты таңдаңыз.", 400)

    # --- 2. Загружаем places.json и фильтруем по интересам -----------------
    all_places = load_places()
    places = [p for p in all_places if p["category"] in interests]
    # Словарь id -> объект: по нему быстро проверяем ответ ИИ
    places_by_id = {p["id"]: p for p in places}

    # --- 3. Проверяем наличие ключа ----------------------------------------
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return error(ERR_NO_KEY, 500)

    # В ИИ отправляем только нужные поля (координаты ИИ не нужны)
    places_for_ai = [
        {
            "id": p["id"],
            "name_kz": p["name_kz"],
            "city": p["city"],
            "category": p["category"],
            "description_kz": p["description_kz"],
            "visit_hours": p["visit_hours"],
        }
        for p in places
    ]

    user_message = (
        f"Шығатын қала: {city}\n"
        f"Күн саны: {days}\n"
        f"Жас тобы: {age_group}\n"
        f"Қызығушылық: {', '.join(ALLOWED_INTERESTS[i] for i in interests)}\n\n"
        f"places.json тізімі:\n{json.dumps(places_for_ai, ensure_ascii=False)}"
    )

    # --- 4. Вызываем OpenAI -------------------------------------------------
    try:
        client = OpenAI(api_key=api_key, timeout=60)
        completion = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            # Просим модель вернуть строго валидный JSON
            response_format={"type": "json_object"},
            temperature=0.4,
        )
        ai_data = json.loads(completion.choices[0].message.content)
    except AuthenticationError:
        return error(ERR_AUTH, 401)
    except RateLimitError:
        return error(ERR_RATE, 429)
    except APITimeoutError:
        return error(ERR_TIMEOUT, 504)
    except APIConnectionError:
        return error(ERR_NETWORK, 503)
    except (BadRequestError, OpenAIError):
        return error(ERR_OTHER, 502)
    except (json.JSONDecodeError, TypeError, IndexError):
        # ИИ вернул не JSON или пустой ответ
        return error(ERR_BAD_ANSWER, 502)

    # --- 5. Проверяем ответ ИИ: отбрасываем неизвестные id -----------------
    if not isinstance(ai_data, dict):
        return error(ERR_BAD_ANSWER, 502)

    clean_days = []
    used_ids = set()  # один объект не должен повторяться в маршруте
    raw_days = ai_data.get("days")
    for raw_day in raw_days if isinstance(raw_days, list) else []:
        if not isinstance(raw_day, dict):
            continue
        clean_stops = []
        raw_stops = raw_day.get("stops")
        for raw_stop in raw_stops if isinstance(raw_stops, list) else []:
            if not isinstance(raw_stop, dict):
                continue
            place = places_by_id.get(raw_stop.get("id"))
            # Неизвестный id (ИИ выдумал объект) или повтор — пропускаем
            if place is None or place["id"] in used_ids:
                continue
            used_ids.add(place["id"])
            rules = raw_stop.get("rules")
            clean_stops.append({
                "id": place["id"],
                # Данные объекта берём из places.json, а не из ответа ИИ —
                # так название и координаты всегда настоящие
                "name_kz": place["name_kz"],
                "city": place["city"],
                "lat": place["lat"],
                "lng": place["lng"],
                "category": place["category"],
                "description_kz": place["description_kz"],
                "visit_hours": place["visit_hours"],
                # Поля, которые написал ИИ
                "time": str(raw_stop.get("time", "")),
                "note": str(raw_stop.get("note", "")),
                "rules": [str(r) for r in rules] if isinstance(rules, list) else [],
            })
        if clean_stops:
            clean_days.append({"day": len(clean_days) + 1, "stops": clean_stops})

    # Если после фильтрации ничего не осталось — сообщаем об этом
    if not clean_days:
        return error(ERR_EMPTY, 502)

    return jsonify({
        "title": str(ai_data.get("title", "Есте сақтау жолы")),
        "days": clean_days,
        "why_important": str(ai_data.get("why_important", "")),
    })


if __name__ == "__main__":
    # Локальный запуск: http://127.0.0.1:5000
    app.run(host="127.0.0.1", port=5000, debug=True)
