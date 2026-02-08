# File: main.py — основной бот (Telegram-бот для загрузки PDF ТЗ, подтверждения "только 100%",
# извлечения требований, (MVP) попытки подбора и формирования PDF-отчёта + история до 100 результатов).

import asyncio
import json
import logging
import os
import re
import shutil
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from telegram import (
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Update,
)
from telegram.constants import ChatAction, ParseMode
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

# -----------------------------
# 0) Настройки/пути/константы
# -----------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
FILES_DIR = DATA_DIR / "files"
REPORTS_DIR = DATA_DIR / "reports"
HISTORY_PATH = DATA_DIR / "history.json"

MAX_HISTORY = 100  # хранить до 100 последних результатов (по ТЗ)

CONFIRM_CB = "confirm_100"
CANCEL_CB = "cancel"

HELP_TEXT = (
    "Я бот для подбора товаров/моделей по ТЗ (PDF) со строгим правилом: *только 100% соответствие*.\n\n"
    "Как пользоваться:\n"
    "1) Отправьте мне PDF-файл технического задания.\n"
    "2) Подтвердите, что нужен подбор *только при 100% соответствии*.\n"
    "3) Дождитесь обработки и получите PDF-отчёт.\n\n"
    "Команды:\n"
    "/start — инструкция\n"
    "/history — показать последние результаты\n"
    "/help — справка\n"
)

# -----------------------------
# 1) Простейший .env loader (без зависимостей)
# -----------------------------

def load_dotenv(dotenv_path: Path) -> Dict[str, str]:
    """
    Минимальный парсер .env:
    - KEY=VALUE
    - поддерживает кавычки "..." или '...'
    - игнорирует пустые строки и комментарии
    """
    env: Dict[str, str] = {}
    if not dotenv_path.exists():
        return env

    for raw_line in dotenv_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip()
        if not key:
            continue
        # снять кавычки
        if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
            val = val[1:-1]
        env[key] = val
    return env


def get_required_env() -> Tuple[str, str]:
    """
    Читает .env рядом с main.py и возвращает (TELEGRAM_TOKEN, ADMIN_CHAT_ID).
    Если переменных нет — печатает понятное сообщение и завершает программу.
    """
    dotenv = load_dotenv(BASE_DIR / ".env")

    token = dotenv.get("TELEGRAM_TOKEN") or os.environ.get("TELEGRAM_TOKEN")
    admin_chat_id = dotenv.get("ADMIN_CHAT_ID") or os.environ.get("ADMIN_CHAT_ID")

    missing = []
    if not token:
        missing.append("TELEGRAM_TOKEN")
    if not admin_chat_id:
        missing.append("ADMIN_CHAT_ID")

    if missing:
        print(
            "Ошибка: не найдены обязательные переменные окружения.\n"
            "Добавьте в файл .env рядом с main.py строки:\n"
            "TELEGRAM_TOKEN=ваш_токен\n"
            "ADMIN_CHAT_ID=ваш_chat_id\n"
            f"Не хватает: {', '.join(missing)}"
        )
        sys.exit(1)

    # базовая валидация chat_id
    admin_chat_id = str(admin_chat_id).strip()
    if not re.fullmatch(r"-?\d+", admin_chat_id):
        print(
            "Ошибка: ADMIN_CHAT_ID должен быть числом (например 123456789 или -1001234567890).\n"
            "Проверьте значение в .env."
        )
        sys.exit(1)

    return token.strip(), admin_chat_id


# -----------------------------
# 2) Хранилище истории (json, до 100)
# -----------------------------

def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FILES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    if not HISTORY_PATH.exists():
        HISTORY_PATH.write_text("[]", encoding="utf-8")


def load_history() -> List[Dict[str, Any]]:
    try:
        data = json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
        return []
    except Exception:
        return []


def save_history(items: List[Dict[str, Any]]) -> None:
    # ограничение до MAX_HISTORY, вытесняя старые
    items = items[-MAX_HISTORY:]
    HISTORY_PATH.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")


def add_history_record(record: Dict[str, Any]) -> None:
    items = load_history()
    items.append(record)
    save_history(items)


# -----------------------------
# 3) Модели данных (MVP)
# -----------------------------

@dataclass
class PositionRequirement:
    position_name: str
    raw_requirements_text: str


@dataclass
class CandidateProduct:
    title: str
    url: str
    price: Optional[str] = None
    availability: Optional[str] = None
    supplier: Optional[str] = None
    characteristics: Optional[Dict[str, str]] = None
    sources: Optional[Dict[str, str]] = None


@dataclass
class MatchResult:
    position_name: str
    requirements: Dict[str, str]
    matched: List[CandidateProduct]
    not_found_reasons: List[str]


# -----------------------------
# 4) Извлечение текста из PDF
# -----------------------------

def extract_text_from_pdf(pdf_path: Path) -> str:
    """
    Пытается извлечь текст из PDF.
    Использует PyPDF2, если установлен. Если нет — возвращает пустую строку.
    """
    try:
        from PyPDF2 import PdfReader  # type: ignore
    except Exception:
        return ""

    try:
        reader = PdfReader(str(pdf_path))
        texts = []
        for page in reader.pages:
            t = page.extract_text() or ""
            texts.append(t)
        return "\n".join(texts).strip()
    except Exception:
        return ""


# -----------------------------
# 5) Извлечение "позиций" и "характеристик" из текста (MVP, без магии)
# -----------------------------

def split_positions(text: str) -> List[PositionRequirement]:
    """
    MVP-разделение на позиции.
    Реальный парсинг ТЗ — отдельная большая задача. Здесь:
    - если встречаем маркеры "Позиция" / "Лот" / "Наименование" — пытаемся сегментировать
    - иначе считаем, что в PDF одна позиция целиком.
    """
    cleaned = re.sub(r"\r\n?", "\n", text).strip()
    if not cleaned:
        return [PositionRequirement(position_name="Позиция 1", raw_requirements_text="")]

    # Попытка найти заголовки позиций
    # Примерные триггеры: "Позиция 1", "Лот 2", "Наименование товара:"
    pattern = re.compile(r"(?im)^(позици[яи]\s*\d+|лот\s*\d+|наименование\s+товара\s*:)\s*", re.MULTILINE)
    matches = list(pattern.finditer(cleaned))

    if len(matches) <= 1:
        return [PositionRequirement(position_name="Позиция 1", raw_requirements_text=cleaned)]

    parts: List[PositionRequirement] = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(cleaned)
        chunk = cleaned[start:end].strip()

        # имя позиции — первая строка
        first_line = chunk.split("\n", 1)[0].strip()
        pos_name = first_line[:80] if first_line else f"Позиция {i+1}"
        parts.append(PositionRequirement(position_name=pos_name, raw_requirements_text=chunk))

    return parts


def parse_requirements_kv(raw_text: str) -> Dict[str, str]:
    """
    MVP-извлечение требований как ключ:значение.
    - берём строки вида "Параметр: значение"
    - или "Параметр - значение"
    - остальное складываем в "Текст"
    """
    req: Dict[str, str] = {}
    lines = [ln.strip() for ln in raw_text.split("\n") if ln.strip()]
    if not lines:
        return {"Текст": ""}

    kv_re = re.compile(r"^(.{2,80}?)[\:\-]\s+(.{1,200})$")
    free_text: List[str] = []

    for ln in lines:
        m = kv_re.match(ln)
        if m:
            key = m.group(1).strip()
            val = m.group(2).strip()
            # избегаем дублей ключей
            if key in req:
                req[key] = req[key] + " | " + val
            else:
                req[key] = val
        else:
            free_text.append(ln)

    if free_text and "Текст" not in req:
        req["Текст"] = "\n".join(free_text)[:5000]

    return req


# -----------------------------
# 6) Поиск кандидатов (MVP-заглушка)
# -----------------------------

async def find_candidates_for_position(_position: PositionRequirement) -> List[CandidateProduct]:
    """
    MVP-заглушка. По ТЗ нужен веб-поиск и извлечение характеристик/источников.
    В этом main.py мы не делаем реальный парсинг интернета (нужны отдельные модули, правила, антибан и т.д.).
    Возвращаем пустой список, чтобы бот корректно выдавал "НЕ НАЙДЕНО".
    """
    await asyncio.sleep(0)  # чтобы функция была действительно async
    return []


def strict_100_percent_match(_requirements: Dict[str, str], _candidate: CandidateProduct) -> Tuple[bool, List[str]]:
    """
    Строгая проверка "100%".
    Для реальной логики нужно иметь:
    - нормализованные характеристики кандидата
    - источники подтверждения по каждой характеристике
    Сейчас (MVP) всегда False, т.к. кандидатов нет.
    """
    return False, ["MVP: модуль сопоставления не реализован без извлечения характеристик из источников."]


async def process_positions(positions: List[PositionRequirement]) -> List[MatchResult]:
    results: List[MatchResult] = []

    for pos in positions:
        requirements = parse_requirements_kv(pos.raw_requirements_text)
        candidates = await find_candidates_for_position(pos)

        matched: List[CandidateProduct] = []
        not_found_reasons: List[str] = []

        if not candidates:
            not_found_reasons.append("НЕ НАЙДЕНО: не удалось получить кандидатов товаров/моделей.")
        else:
            for cand in candidates:
                ok, reasons = strict_100_percent_match(requirements, cand)
                if ok:
                    matched.append(cand)
                else:
                    # причины несовпадения — можно агрегировать
                    not_found_reasons.extend(reasons[:3])

            if not matched:
                not_found_reasons.append("НЕ НАЙДЕНО: нет товаров, которые соответствуют на 100% по всем характеристикам.")

        results.append(
            MatchResult(
                position_name=pos.position_name,
                requirements=requirements,
                matched=matched,
                not_found_reasons=not_found_reasons[:20],
            )
        )

    return results


# -----------------------------
# 7) Генерация PDF отчёта (если доступен reportlab), иначе TXT
# -----------------------------

def build_report_text(results: List[MatchResult]) -> str:
    lines: List[str] = []
    lines.append("Отчёт подбора по ТЗ (строго: только 100% соответствие)")
    lines.append(f"Дата/время: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 70)

    for idx, r in enumerate(results, 1):
        lines.append(f"\nПозиция {idx}: {r.position_name}")
        lines.append("-" * 70)

        lines.append("Требования (извлечённые):")
        for k, v in list(r.requirements.items())[:200]:
            lines.append(f" - {k}: {v}")

        if r.matched:
            lines.append("\nПодходящие товары/модели (100%):")
            for p in r.matched:
                lines.append(f" * {p.title}")
                lines.append(f"   URL: {p.url}")
                if p.price:
                    lines.append(f"   Цена: {p.price}")
                if p.availability:
                    lines.append(f"   Наличие: {p.availability}")
                if p.supplier:
                    lines.append(f"   Поставщик: {p.supplier}")
        else:
            lines.append("\nРезультат: НЕ НАЙДЕНО")
            if r.not_found_reasons:
                lines.append("Причины/блокирующие параметры (MVP):")
                for reason in r.not_found_reasons:
                    lines.append(f" - {reason}")

    lines.append("\n" + "=" * 70)
    lines.append("Примечание: Это MVP-скелет. Реальный веб-поиск/извлечение характеристик/источников нужно реализовать отдельными модулями.")
    return "\n".join(lines)


def save_pdf_or_fallback_txt(report_text: str, out_base_path: Path) -> Tuple[Path, str]:
    """
    Возвращает (путь_к_файлу, mime_type).
    Пытаемся сделать PDF через reportlab, если нет — сохраняем .txt
    """
    try:
        from reportlab.lib.pagesizes import A4  # type: ignore
        from reportlab.lib.units import mm  # type: ignore
        from reportlab.pdfbase import pdfmetrics  # type: ignore
        from reportlab.pdfbase.ttfonts import TTFont  # type: ignore
        from reportlab.pdfgen import canvas  # type: ignore

        # Регистрация шрифта с кириллицей (если есть в системе)
        # Пытаемся несколько вариантов, иначе стандартный (может не вывести кириллицу корректно).
        font_name = "Helvetica"
        for font_path in [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/Library/Fonts/Arial Unicode.ttf",
            "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
            str(BASE_DIR / "DejaVuSans.ttf"),
        ]:
            if os.path.exists(font_path):
                try:
                    pdfmetrics.registerFont(TTFont("DejaVuSans", font_path))
                    font_name = "DejaVuSans"
                    break
                except Exception:
                    pass

        pdf_path = out_base_path.with_suffix(".pdf")
        c = canvas.Canvas(str(pdf_path), pagesize=A4)
        width, height = A4

        c.setTitle("Tender Match Report (MVP)")
        c.setFont(font_name, 11)

        margin_x = 15 * mm
        margin_y = 15 * mm
        y = height - margin_y

        # Простейшая разметка строк
        for line in report_text.split("\n"):
            if y < margin_y:
                c.showPage()
                c.setFont(font_name, 11)
                y = height - margin_y

            # длинные строки режем
            safe = line
            max_chars = 110
            while len(safe) > max_chars:
                chunk = safe[:max_chars]
                c.drawString(margin_x, y, chunk)
                y -= 6 * mm
                safe = safe[max_chars:]
                if y < margin_y:
                    c.showPage()
                    c.setFont(font_name, 11)
                    y = height - margin_y
            c.drawString(margin_x, y, safe)
            y -= 6 * mm

        c.save()
        return pdf_path, "application/pdf"
    except Exception:
        txt_path = out_base_path.with_suffix(".txt")
        txt_path.write_text(report_text, encoding="utf-8")
        return txt_path, "text/plain"


# -----------------------------
# 8) Telegram-логика (диалог/сессии)
# -----------------------------

def session_key(chat_id: int) -> str:
    return f"session:{chat_id}"


def require_confirm_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [InlineKeyboardButton("✅ Подтверждаю: только 100% соответствие", callback_data=CONFIRM_CB)],
            [InlineKeyboardButton("❌ Отмена", callback_data=CANCEL_CB)],
        ]
    )


async def safe_notify_admin(context: ContextTypes.DEFAULT_TYPE, admin_chat_id: str, text: str) -> None:
    try:
        await context.bot.send_message(chat_id=int(admin_chat_id), text=text)
    except Exception:
        # не падаем из-за уведомления админу
        pass


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(HELP_TEXT, parse_mode=ParseMode.MARKDOWN)
    # сброс сессии
    context.user_data.pop(session_key(update.effective_chat.id), None)


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(HELP_TEXT, parse_mode=ParseMode.MARKDOWN)


async def cmd_history(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    items = load_history()
    if not items:
        await update.message.reply_text("История пуста. Отправьте PDF с ТЗ, чтобы получить первый результат.")
        return

    # показываем последние 10
    last = items[-10:]
    lines = ["Последние результаты (до 10):\n"]
    for it in reversed(last):
        ts = it.get("created_at", "")
        rid = it.get("result_id", "")
        fname = it.get("report_file", "")
        pos_cnt = it.get("positions_count", "")
        lines.append(f"• {ts} | result_id={rid} | позиций={pos_cnt} | файл={fname}")
    await update.message.reply_text("\n".join(lines))


async def on_document(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    msg = update.message
    if not msg or not msg.document:
        return

    doc = msg.document
    filename = doc.file_name or "document"
    ext = (Path(filename).suffix or "").lower()

    if ext != ".pdf":
        await msg.reply_text("Формат файла не поддерживается (поддерживается только PDF).")
        return

    await msg.reply_text("Файл получен. Перед обработкой подтвердите правило: *только 100% соответствие*.",
                         parse_mode=ParseMode.MARKDOWN,
                         reply_markup=require_confirm_keyboard())

    # скачиваем файл в data/files
    ensure_dirs()
    file = await context.bot.get_file(doc.file_id)
    result_id = uuid.uuid4().hex[:12]
    saved_pdf = FILES_DIR / f"{result_id}.pdf"
    try:
        await file.download_to_drive(custom_path=str(saved_pdf))
    except Exception:
        await msg.reply_text("Файл не загружен или повреждён.")
        return

    # сохраняем в сессии
    context.user_data[session_key(update.effective_chat.id)] = {
        "result_id": result_id,
        "pdf_path": str(saved_pdf),
        "confirmed_100": False,
        "original_filename": filename,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


async def on_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    if not query:
        return
    await query.answer()

    chat_id = update.effective_chat.id
    skey = session_key(chat_id)
    sess = context.user_data.get(skey)

    if query.data == CANCEL_CB:
        context.user_data.pop(skey, None)
        await query.edit_message_text("Ок, отменено. Можете отправить новый PDF.")
        return

    if query.data == CONFIRM_CB:
        if not sess:
            await query.edit_message_text("Сессия не найдена. Пожалуйста, отправьте PDF заново.")
            return

        sess["confirmed_100"] = True
        context.user_data[skey] = sess
        await query.edit_message_text("✅ Подтверждение принято. Начинаю обработку...")

        # запуск обработки
        await run_processing_pipeline(update, context, sess)
        return


async def run_processing_pipeline(update: Update, context: ContextTypes.DEFAULT_TYPE, sess: Dict[str, Any]) -> None:
    """
    Полный пайплайн:
    - проверка подтверждения
    - извлечение текста
    - выделение позиций
    - поиск/проверка (MVP)
    - сбор отчёта
    - генерация PDF
    - отправка
    - запись в историю (до 100)
    """
    chat_id = update.effective_chat.id
    result_id = sess["result_id"]
    pdf_path = Path(sess["pdf_path"])

    if not sess.get("confirmed_100"):
        await context.bot.send_message(chat_id=chat_id, text="Нужно подтверждение *только 100% соответствие*.", parse_mode=ParseMode.MARKDOWN)
        return

    # статус
    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)

    text = extract_text_from_pdf(pdf_path)
    if not text:
        await context.bot.send_message(
            chat_id=chat_id,
            text=(
                "Не удалось извлечь текст из PDF.\n\n"
                "Возможные причины:\n"
                "• PDF состоит из сканов/картинок (нужен OCR)\n"
                "• В окружении не установлен PyPDF2\n\n"
                "MVP сейчас работает только с PDF, где есть извлекаемый текст."
            ),
        )
        return

    positions = split_positions(text)
    await context.bot.send_message(chat_id=chat_id, text=f"Найдено позиций: {len(positions)}. Начинаю подбор по каждой...")

    # обработка
    try:
        await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
        results = await process_positions(positions)
    except Exception as e:
        await context.bot.send_message(chat_id=chat_id, text="Нет доступа к обработке: перегрузка/очередь.")
        await safe_notify_admin(context, ADMIN_CHAT_ID, f"[ERROR] processing failed: {e!r}")
        return

    report_text = build_report_text(results)
    out_base = REPORTS_DIR / f"{result_id}_report"
    report_path, mime = save_pdf_or_fallback_txt(report_text, out_base)

    # отправка отчёта
    try:
        await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.UPLOAD_DOCUMENT)
        with report_path.open("rb") as f:
            if mime == "application/pdf":
                await context.bot.send_document(chat_id=chat_id, document=f, filename=report_path.name, caption="Готово. Отчёт (PDF).")
            else:
                await context.bot.send_document(chat_id=chat_id, document=f, filename=report_path.name, caption="Готово. PDF не удалось собрать — отправляю TXT (MVP).")
    except Exception as e:
        await context.bot.send_message(chat_id=chat_id, text="Ошибка при отправке результата.")
        await safe_notify_admin(context, ADMIN_CHAT_ID, f"[ERROR] sending report failed: {e!r}")
        return

    # история
    add_history_record(
        {
            "result_id": result_id,
            "created_at": sess.get("created_at"),
            "chat_id": chat_id,
            "original_filename": sess.get("original_filename"),
            "source_pdf": str(pdf_path),
            "report_file": str(report_path),
            "positions_count": len(positions),
        }
    )

    # чистим сессию
    context.user_data.pop(session_key(chat_id), None)

    await context.bot.send_message(chat_id=chat_id, text="Можете отправить новое ТЗ (PDF), если нужно.")


# -----------------------------
# 9) main()
# -----------------------------

def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        stream=sys.stdout,
    )


TELEGRAM_TOKEN, ADMIN_CHAT_ID = get_required_env()


def main() -> None:
    setup_logging()
    ensure_dirs()

    app = Application.builder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("history", cmd_history))

    app.add_handler(CallbackQueryHandler(on_callback))
    app.add_handler(MessageHandler(filters.Document.ALL, on_document))

    # Запуск (long polling)
    app.run_polling(close_loop=False)


if __name__ == "__main__":
    main()