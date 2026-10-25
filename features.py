"""
Общая логика: парсинг/нормализация полей OCR, проверка валидности паспорта,
справочник подразделений, TD3-парсер MRZ и извлечение признаков.

Используется и в train.py, optuna_search.py и в postprocess.py.
"""
import csv
import re
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path

import numpy as np

TARGET_DATE = date(2026, 10, 1)
MIN_YEAR, MAX_YEAR = 1900, 2030

FIELD_NAMES = [
    "Last_name_ru", "First_name_ru", "Middle_name_ru", "Sex_ru",
    "Birth_date", "Birth_place_ru", "Issue_date",
    "Issue_organisation_code", "Licence_number", "Issue_organization_ru",
]


# ---------------------------------------------------------------- paths

def find_data_dir(explicit=None):
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    here = Path(__file__).resolve().parent
    candidates += [
        here / "data", here.parent / "data",
        Path.cwd() / "data", Path.cwd(),
        here, here.parent,
    ]
    seen = set()
    for c in candidates:
        c = c.resolve()
        if c in seen:
            continue
        seen.add(c)
        if (c / "train_predictions.jsonl").exists() \
                or (c / "test_predictions.jsonl").exists():
            return c
    return None


def describe_dir(path):
    p = Path(path)
    print(f"[paths] {p} exists={p.exists()}")
    if not p.exists():
        return
    for item in sorted(p.iterdir()):
        tag = "DIR " if item.is_dir() else "FILE"
        print(f"[paths]   {tag} {item.name}")


# ---------------------------------------------------------------- dates

_DATE_FMTS = ("%d.%m.%Y", "%d.%m.%y", "%d-%m-%Y", "%d-%m-%y",
              "%d/%m/%Y", "%Y-%m-%d", "%Y/%m/%d")

def parse_date(s):
    """Разбирает дату в известных форматах, не извлекая её из произвольного текста."""
    if not s:
        return None
    s = normalize_spaces(str(s))
    if not s:
        return None

    for fmt in _DATE_FMTS:
        try:
            d = datetime.strptime(s, fmt).date()
        except ValueError:
            continue
        if MIN_YEAR <= d.year <= MAX_YEAR:
            return d

    # Допускаем OCR-пробелы/разделители, но вся строка должна быть датой.
    match = re.fullmatch(r"(\d{1,2})\D+(\d{1,2})\D+(\d{2,4})", s)
    if not match:
        match = re.fullmatch(r"(\d{4})\D+(\d{1,2})\D+(\d{1,2})", s)
        if match:
            y, mo, d = map(int, match.groups())
        else:
            return None
    else:
        d, mo, y = map(int, match.groups())
        if y < 100:
            y += 1900 if y > 30 else 2000

    if not (MIN_YEAR <= y <= MAX_YEAR):
        return None
    try:
        return date(y, mo, d)
    except ValueError:
        return None

def format_date(d):
    return d.strftime("%d.%m.%Y") if d else ""


# ---------------------------------------------------------------- corrector

def normalize_spaces(s):
    return re.sub(r"\s+", " ", (s or "")).strip()

def correct_field(fname, text):
    s = normalize_spaces(text)
    if not s:
        return ""
    if fname in ("Last_name_ru", "First_name_ru", "Middle_name_ru"):
        return s.upper()
    if fname == "Sex_ru":
        up = s.upper()
        if up.startswith("МУЖ"):
            return "МУЖ."
        if up.startswith("ЖЕН"):
            return "ЖЕН."
        return up
    if fname in ("Birth_date", "Issue_date"):
        d = parse_date(s)
        return format_date(d) if d else s
    if fname == "Issue_organisation_code":
        d = re.sub(r"\D", "", s)
        return f"{d[:3]}-{d[3:6]}" if len(d) >= 6 else s
    if fname == "Licence_number":
        d = re.sub(r"\D", "", s)
        if len(d) == 10:
            return f"{d[:2]} {d[2:4]} {d[4:]}"
        return s
    if fname in ("Birth_place_ru", "Issue_organization_ru"):
        return s.upper()
    return s


# ---------------------------------------------------------------- validity

def _add_years(d, years):
    """Календарное прибавление лет; 29 февраля переходит на 28 февраля."""
    try:
        return d.replace(year=d.year + years)
    except ValueError:
        return d.replace(year=d.year + years, day=28)


def passport_valid(fields, target_date=TARGET_DATE):
    """Проверяет срок действия внутреннего паспорта по возрасту при выдаче."""
    issue = parse_date(fields.get("Issue_date", ""))
    birth = parse_date(fields.get("Birth_date", ""))
    if issue is None or birth is None:
        return False
    if birth > issue or issue > target_date:
        return False

    age_at_issue = issue.year - birth.year
    if (issue.month, issue.day) < (birth.month, birth.day):
        age_at_issue -= 1

    # Паспорт, выданный с 45 лет, считается бессрочным.
    if age_at_issue >= 45:
        return True
    if age_at_issue < 14:
        return False

    # Для паспорта, выданного до 20 лет, срок заканчивается в 20 лет;
    # для выданного с 20 до 45 лет — в 45 лет.
    expiry = _add_years(birth, 20 if age_at_issue < 20 else 45)
    return expiry > target_date


# ---------------------------------------------------------------- viz

def get_viz_text(fdata):
    if not fdata:
        return "", []
    lines = fdata.get("lines") or []
    parts = [normalize_spaces(ln.get("text") or "") for ln in lines]
    ccs = []
    for ln in lines:
        ccs.extend(ln.get("char_conf") or [])
    return normalize_spaces(" ".join(p for p in parts if p)), ccs


# ---------------------------------------------------------------- transliteration

# Согласно встречающимся в РФ-паспортах вариантам:
_TRANSLIT = {
    "А": "A", "Б": "B", "В": "V", "Г": "G", "Д": "D", "Е": "E", "Ё": "E",
    "Ж": "J", "З": "Z", "И": "I", "Й": "Q", "К": "K", "Л": "L", "М": "M",
    "Н": "N", "О": "O", "П": "P", "Р": "R", "С": "S", "Т": "T", "У": "U",
    "Ф": "F", "Х": "H", "Ц": "C", "Ч": "3", "Ш": "4", "Щ": "W",
    "Ъ": "X", "Ы": "Y", "Ь": "9", "Э": "6", "Ю": "7", "Я": "8",
}

def _translit(s):
    return "".join(_TRANSLIT.get(ch, ch) for ch in (s or "").upper())


def _char_overlap(a, b):
    """Multiset char-overlap, устойчив к OCR-заменам одной буквы."""
    if not a or not b:
        return 0.0
    ca, cb = {}, {}
    for ch in a:
        ca[ch] = ca.get(ch, 0) + 1
    for ch in b:
        cb[ch] = cb.get(ch, 0) + 1
    common = sum(min(ca[ch], cb.get(ch, 0)) for ch in ca)
    return common / max(len(a), len(b))


def _is_subsequence(needle, haystack):
    if not needle:
        return False
    j = 0
    for ch in haystack:
        if ch == needle[j]:
            j += 1
            if j == len(needle):
                return True
    return False


# ---------------------------------------------------------------- MRZ parser

_MRZ_WEIGHTS = (7, 3, 1)

def _mrz_char_value(ch):
    if ch == "<":
        return 0
    if "0" <= ch <= "9":
        return int(ch)
    if "A" <= ch <= "Z":
        return ord(ch) - ord("A") + 10
    return None

def _mrz_check_digit(chars):
    total = 0
    for i, ch in enumerate(chars):
        v = _mrz_char_value(ch)
        if v is None:
            return None
        total += v * _MRZ_WEIGHTS[i % 3]
    return total % 10

def _clean_mrz_line(text, length=44):
    s = re.sub(r"[^A-Z0-9<]", "", str(text or "").upper())
    if len(s) > length:
        return s[:length]
    return s + "<" * (length - len(s))


def _parse_yymmdd(s, kind="birth", reference_date=TARGET_DATE):
    """Разбирает дату MRZ с выбором века по контексту поля."""
    if not s or len(s) != 6 or not s.isdigit():
        return None
    yy, mm, dd = int(s[:2]), int(s[2:4]), int(s[4:])
    candidates = []
    for year in (1900 + yy, 2000 + yy):
        if MIN_YEAR <= year <= MAX_YEAR:
            try:
                candidates.append(date(year, mm, dd))
            except ValueError:
                pass
    if not candidates:
        return None

    if kind == "birth":
        # Выбираем наиболее позднюю дату рождения, которая не в будущем
        # и соответствует реалистичному возрасту (0–120 лет).
        plausible = [
            d for d in candidates
            if d <= reference_date and d >= date(reference_date.year - 120, reference_date.month, min(reference_date.day, 28))
        ]
        if plausible:
            return max(plausible)
        past = [d for d in candidates if d <= reference_date]
        return max(past) if past else None

    if kind == "expiry":
        # Дата окончания обычно близка к дате проверки; допускаем небольшой
        # срок просрочки, чтобы не терять информацию в признаках.
        future = [d for d in candidates if d >= reference_date]
        if future:
            return min(future)
        return max(candidates)

    return candidates[0]

def parse_mrz(mrz):
    """TD3 (2×44) с адаптацией под РФ-паспорт."""
    if not mrz:
        return None
    lines = mrz.get("lines") or []
    if len(lines) < 2:
        return None

    raw0 = re.sub(r"[^A-Z0-9<]", "", str((lines[0] or {}).get("text") or "").upper())
    raw1 = re.sub(r"[^A-Z0-9<]", "", str((lines[1] or {}).get("text") or "").upper())
    # TD3 имеет две длинные строки. Не превращаем короткий произвольный текст
    # в MRZ простым дополнением символами "<".
    if len(raw0) < 30 or len(raw1) < 30:
        return None

    l0 = _clean_mrz_line(raw0)
    l1 = _clean_mrz_line(raw1)
    if l1.startswith("P") and not l0.startswith("P"):
        l0, l1 = l1, l0

    doc_type = l0[0]
    issuing_country = l0[2:5]
    if doc_type not in ("P", "V", "I", "A", "C"):
        return None
    if not re.fullmatch(r"[A-Z<]{3}", issuing_country):
        return None
    name_part = l0[5:]
    if "<<" in name_part:
        surname, given = name_part.split("<<", 1)
    else:
        surname, given = name_part, ""
    surname = surname.replace("<", " ").strip()
    given = re.sub(r"<+", " ", given).strip()

    number_field    = l1[0:9]
    number_check    = l1[9]
    nationality     = l1[10:13]
    birth_field     = l1[13:19]
    birth_check     = l1[19]
    sex_field       = l1[20]
    expiry_field    = l1[21:27]
    expiry_check    = l1[27]
    personal_field  = l1[28:42]
    personal_check  = l1[42]
    composite_check = l1[43]

    checks = {}

    def _check(name, field, cd):
        if not cd.isdigit():
            return
        expected = _mrz_check_digit(field)
        checks[name] = (expected is not None and expected == int(cd))

    _check("number",  number_field,  number_check)
    _check("birth",   birth_field,   birth_check)
    if expiry_field.strip("<"):
        _check("expiry", expiry_field, expiry_check)
    _check("personal", personal_field, personal_check)

    if composite_check.isdigit():
        composite_str = (
            number_field + number_check
            + birth_field + birth_check
            + expiry_field + expiry_check
            + personal_field + personal_check
        )
        expected = _mrz_check_digit(composite_str)
        checks["composite"] = (expected is not None
                               and expected == int(composite_check))

    return {
        "doc_type":        doc_type,
        "issuing_country": issuing_country,
        "surname":         surname,
        "given_names":     given,
        "number":          number_field.replace("<", ""),
        "nationality":     nationality,
        "birth_field":     birth_field,
        "birth_date":      _parse_yymmdd(birth_field, kind="birth"),
        "sex":             {"M": "МУЖ.", "F": "ЖЕН."}.get(sex_field),
        "expiry_field":    expiry_field,
        "expiry_date":     _parse_yymmdd(expiry_field, kind="expiry"),
        "personal_number": personal_field.replace("<", ""),
        "checks":          checks,
    }


# ---------------------------------------------------------------- subdivisions

def load_subdivisions(path):
    path = Path(path)
    if not path.exists():
        return {}
    with open(path, encoding="utf-8-sig") as f:
        first = f.readline()
    delim = "\t" if "\t" in first else ","
    out = defaultdict(list)
    with open(path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, delimiter=delim)
        cols = reader.fieldnames or []
        code_col = next((c for c in cols if c.strip().lower() == "code"), None)
        name_col = next((c for c in cols if c.strip().lower() == "name"), None)
        if code_col is None or name_col is None:
            return {}
        for row in reader:
            code = (row.get(code_col) or "").strip()
            name = (row.get(name_col) or "").strip()
            if not code:
                continue
            digits = re.sub(r"\D", "", code)
            key = f"{digits[:3]}-{digits[3:6]}" if len(digits) >= 6 else code
            if name:
                out[key].append(name)
    return dict(out)


_TOKEN_RE = re.compile(r"[А-ЯЁA-Z]+")
_STOP_TOKENS = {
    "РОССИИ", "РОССИЙСКОЙ", "РОССИЯ", "ФЕДЕРАЦИИ", "ФЕДЕРАЦИЯ",
    "МВД", "ОВД", "ОМВД", "УВД", "УМВД", "ГУ", "ГУВД", "МО",
    "ФМС", "УФМС", "ОФМС",
    "ОТДЕЛ", "ОТДЕЛЕНИЕ", "ОТДЕЛОМ", "ОТДЕЛЕНИЕМ", "ОТДЕЛЕНИЯ",
    "УПРАВЛЕНИЕ", "УПРАВЛЕНИЕМ", "УПРАВЛЕНИЯ",
    "МИГРАЦИИ", "МИГРАЦИОННОЙ", "ВОПРОСАМ", "ВОПРОСАМИ",
    "ПАСПОРТНО", "ПАСПОРТНОЙ", "ВИЗОВОЙ", "РАБОТЕ",
    "ОБЛАСТИ", "ОБЛАСТЬ", "ОБЛ",
    "РАЙОНЕ", "РАЙОНА", "РАЙОН",
    "КРАЕ", "КРАЯ", "КРАЙ",
    "РЕСПУБЛИКЕ", "РЕСПУБЛИКИ", "РЕСП", "РЕСПУБЛИКА",
    "ГОРОДЕ", "ГОРОДА", "ГОРОД",
    "ПОСЕЛКЕ", "ПОСЕЛКА", "ПОС",
    "СЕЛЕ", "СЕЛА",
    "И", "НА", "ПО", "В", "ОТ", "ДО", "ПРИ",
}

def _geo_tokens(text):
    toks = _TOKEN_RE.findall((text or "").upper())
    return {t for t in toks if len(t) >= 4 and t not in _STOP_TOKENS}


# ---------------------------------------------------------------- features

def _safe_float(value, default=0.0):
    """Безопасное преобразование числовых полей входных данных."""
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return result if np.isfinite(result) else default

def _nan2(v, default=0.0):
    return default if v is None or (isinstance(v, float) and np.isnan(v)) else v

def boost_feature(df, name, times=3, noise=1e-6):
    base = df[name]
    for k in range(times):
        df[f"{name}__dup{k}"] = base + noise
    return df

def extract_features(doc, subdivisions=None):
    q = doc.get("quality") or {}
    viz = doc.get("viz") or {}
    mrz = doc.get("mrz")

    feats = {
        "sharpness": _safe_float(q.get("sharpness")),
        "doc_px":    _safe_float(q.get("doc_px")),
        "glare":     _safe_float(q.get("glare")),
    }
    src = doc.get("source") or ""
    for s in ("web", "mobile", "mobile_v2"):
        feats[f"source_{s}"] = 1.0 if src == s else 0.0

    # ============ MRZ: raw ============
    mrz_txt, mrz_ccs = "", []
    if mrz:
        for ln in mrz.get("lines") or []:
            mrz_txt += ln.get("text") or ""
            mrz_ccs.extend(ln.get("char_conf") or [])
        feats["mrz_present"] = 1.0
        feats["mrz_n_lines"] = float(len(mrz.get("lines") or []))
        feats["mrz_min_cc"]  = _safe_float(min(_safe_float(c) for c in mrz_ccs)) if mrz_ccs else np.nan
    else:
        feats["mrz_present"] = 0.0
        feats["mrz_n_lines"] = 0.0
        feats["mrz_min_cc"]  = np.nan
    mrz_clean = re.sub(r"[^A-Z0-9]", "", mrz_txt.upper())

    # ============ поля viz ============
    corrected = {}
    all_ccs = []
    n_present = 0
    for fname in FIELD_NAMES:
        fdata = viz.get(fname)
        text, ccs = get_viz_text(fdata)
        all_ccs.extend(ccs)

        if not text:
            feats[f"{fname}__present"]  = 0.0
            feats[f"{fname}__n_lines"]  = 0.0
            feats[f"{fname}__conf"]     = _safe_float((fdata or {}).get("conf"))
            feats[f"{fname}__min_cc"]   = np.nan
            feats[f"{fname}__mean_cc"]  = np.nan
            feats[f"{fname}__n_lt_05"]  = np.nan
            feats[f"{fname}__n_lt_07"]  = np.nan
            feats[f"{fname}__n_lt_09"]  = np.nan
            feats[f"{fname}__text_len"] = 0.0
            corrected[fname] = ""
            continue

        n_present += 1
        feats[f"{fname}__present"]  = 1.0
        feats[f"{fname}__n_lines"]  = float(len(fdata.get("lines") or []))
        feats[f"{fname}__conf"]     = _safe_float(fdata.get("conf"))
        feats[f"{fname}__min_cc"]   = _safe_float(min(_safe_float(c) for c in ccs)) if ccs else np.nan
        feats[f"{fname}__mean_cc"]  = _safe_float(np.mean([_safe_float(c) for c in ccs])) if ccs else np.nan
        # feats[f"{fname}__n_lt_05"]  = float(sum(1 for c in ccs if _safe_float(c) < 0.5)) if ccs else np.nan
        # feats[f"{fname}__n_lt_07"]  = float(sum(1 for c in ccs if _safe_float(c) < 0.7)) if ccs else np.nan
        # feats[f"{fname}__n_lt_09"]  = float(sum(1 for c in ccs if _safe_float(c) < 0.9)) if ccs else np.nan
        feats[f"{fname}__text_len"] = float(len(text))
        corrected[fname] = correct_field(fname, text)

    feats["n_fields_present"] = float(n_present)

    if all_ccs:
        # feats["doc_min_cc"]  = _safe_float(min(_safe_float(c) for c in all_ccs))
        feats["doc_mean_cc"] = _safe_float(np.mean([_safe_float(c) for c in all_ccs]))
        # feats["doc_n_lt_05"] = float(sum(1 for c in all_ccs if _safe_float(c) < 0.5))
        # feats["doc_n_lt_07"] = float(sum(1 for c in all_ccs if _safe_float(c) < 0.7))
        # feats["doc_n_lt_09"] = float(sum(1 for c in all_ccs if _safe_float(c) < 0.9))
    else:
        # for k in ("doc_min_cc", "doc_mean_cc",
        #           "doc_n_lt_05", "doc_n_lt_07", "doc_n_lt_09"):
        #     feats[k] = np.nan
        feats["doc_mean_cc"] = np.nan

    feats["passport_valid"] = 1.0 if passport_valid(corrected) else 0.0

    # ============ MRZ: fallback-согласование ============
    birth = parse_date(corrected.get("Birth_date", ""))
    parsed_for_birth = parse_mrz(mrz)
    if birth and parsed_for_birth and parsed_for_birth.get("birth_date"):
        feats["birth_in_mrz"] = 1.0 if birth == parsed_for_birth["birth_date"] else 0.0
    else:
        feats["birth_in_mrz"] = np.nan

    lic = re.sub(r"\D", "", corrected.get("Licence_number", ""))
    if lic and mrz_clean:
        mrz_counts = {}
        for ch in mrz_clean:
            mrz_counts[ch] = mrz_counts.get(ch, 0) + 1
        common = sum(min(mrz_counts.get(ch, 0), lic.count(ch)) for ch in set(lic))
        feats["licence_in_mrz"] = common / len(lic)
    else:
        feats["licence_in_mrz"] = np.nan

    # ============ MRZ: структурный парсер ============

    viz_given_parts = []
    if corrected.get("First_name_ru"):
        viz_given_parts.append(_translit(corrected["First_name_ru"]))
    if corrected.get("Middle_name_ru"):
        viz_given_parts.append(_translit(corrected["Middle_name_ru"]))
    
    parsed = parse_mrz(mrz)

    if parsed:
        checks = parsed["checks"]

        # --- проверки check digits ---
        feats["mrz_parsed"]             = 1.0
        feats["mrz_doc_type_ok"]        = 1.0 if parsed["doc_type"] == "P" else 0.0
        feats["mrz_issuing_country_ok"] = 1.0 if parsed["issuing_country"] == "RUS" else 0.0
        feats["mrz_n_checks"]           = float(len(checks))
        feats["mrz_n_checks_ok"]        = float(sum(checks.values()))
        feats["mrz_all_checks_ok"]      = (
            1.0 if (checks and all(checks.values())) else 0.0
        )
        feats["mrz_birth_check_ok"]     = float(checks.get("birth", False))
        feats["mrz_number_check_ok"]    = float(checks.get("number", False))
        feats["mrz_composite_check_ok"] = float(checks.get("composite", False))

        # --- сравнение: дата рождения ---
        mrz_birth = parsed["birth_date"]
        feats["mrz_birth_eq_viz"] = (
            1.0 if (birth and mrz_birth and birth == mrz_birth) else 0.0
        )

        # --- сравнение: пол ---
        viz_sex = corrected.get("Sex_ru", "")
        mrz_sex = parsed["sex"] or ""
        feats["mrz_sex_eq_viz"] = (
            1.0 if (viz_sex and mrz_sex and viz_sex == mrz_sex) else 0.0
        )

        # --- сравнение: номер (licence vs MRZ number) ---
        mrz_num = parsed["number"] or ""
        if len(lic) >= 6 and len(mrz_num) >= 6:
            feats["mrz_number_tail_eq"] = 1.0 if lic[-6:] == mrz_num[-6:] else 0.0
        else:
            feats["mrz_number_tail_eq"] = 0.0

        if len(lic) >= 3 and len(mrz_num) >= 3:
            feats["mrz_number_head_eq"] = 1.0 if lic[:3] == mrz_num[:3] else 0.0
        else:
            feats["mrz_number_head_eq"] = 0.0

        # Правило РФ-паспорта: MRZ = licence[:3] + licence[4:] (отбрасывается 4-я цифра)
        if len(lic) == 10 and len(mrz_num) == 9:
            expected = lic[:3] + lic[4:]
            feats["mrz_number_full_eq"] = 1.0 if expected == mrz_num else 0.0
        else:
            feats["mrz_number_full_eq"] = 0.0

        # Fallback: MRZ-номер является подпоследовательностью licence
        feats["mrz_number_is_subseq"] = (
            1.0 if (mrz_num and _is_subsequence(mrz_num, lic)) else 0.0
        )

        # --- сравнение: код подразделения (в personal_number MRZ) ---
        code_digits = re.sub(r"\D", "", corrected.get("Issue_organisation_code", ""))
        personal = parsed["personal_number"] or ""
        feats["mrz_code_in_personal"] = (
            1.0 if (code_digits and len(code_digits) == 6
                    and code_digits in personal) else 0.0
        )

        # --- сравнение: ФИО через транслит ---
        viz_surname_t = _translit(corrected.get("Last_name_ru", ""))
        mrz_surname   = parsed["surname"] or ""

        feats["mrz_surname_eq"] = (
            1.0 if (viz_surname_t and mrz_surname
                    and viz_surname_t == mrz_surname) else 0.0
        )
        feats["mrz_surname_initial_eq"] = (
            1.0 if (viz_surname_t and mrz_surname
                    and viz_surname_t[0] == mrz_surname[0]) else 0.0
        )
        feats["mrz_surname_charsim"] = _char_overlap(viz_surname_t, mrz_surname)

        mrz_given_parts = parsed["given_names"].split() if parsed["given_names"] else []

        feats["mrz_given_n_parts"] = float(len(mrz_given_parts))
        feats["viz_given_n_parts"] = float(len(viz_given_parts))
        feats["mrz_given_structure_ok"] = (
            1.0 if len(mrz_given_parts) == len(viz_given_parts) else 0.0
        )

        sims = []
        for i, v in enumerate(viz_given_parts):
            if i < len(mrz_given_parts):
                sims.append(_char_overlap(v, mrz_given_parts[i]))
            else:
                sims.append(0.0)
        
        feats["mrz_firstname_charsim"]  = sims[0] if len(sims) > 0 else 0.0
        feats["mrz_middlename_charsim"] = sims[1] if len(sims) > 1 else 0.0
        
        agree = [
            _nan2(feats["mrz_birth_eq_viz"]),
            _nan2(feats["mrz_sex_eq_viz"]),
            _nan2(feats["mrz_number_full_eq"]),
            _nan2(feats["mrz_surname_eq"]),
            _nan2(feats["mrz_code_in_personal"]),
        ]
        feats["mrz_agreement_sum"]  = sum(agree)
        feats["mrz_agreement_min"]  = min(agree)
        feats["mrz_agreement_mean"] = sum(agree) / len(agree)

        name_sims = [
            _nan2(feats["mrz_surname_charsim"]),
            _nan2(feats["mrz_firstname_charsim"]),
            _nan2(feats["mrz_middlename_charsim"]),
        ]
        feats["name_sim_min"]  = min(name_sims)
        feats["name_sim_mean"] = sum(name_sims) / len(name_sims)
        feats["name_sim_low"]  = float(feats["name_sim_min"] < 0.7)
    else:
        # MRZ нет / не парсится.
        # Факты (parsed, n_checks) = 0, так как их действительно 0.
        # Все сравнительные фичи = np.nan, так как сравнить не с чем.
        feats["mrz_parsed"]             = 0.0
        feats["mrz_doc_type_ok"]        = np.nan
        feats["mrz_issuing_country_ok"] = np.nan
        feats["mrz_n_checks"]           = 0.0
        feats["mrz_n_checks_ok"]        = 0.0
        feats["mrz_all_checks_ok"]      = np.nan
        feats["mrz_birth_check_ok"]     = np.nan
        feats["mrz_number_check_ok"]    = np.nan
        feats["mrz_composite_check_ok"] = np.nan

        feats["mrz_birth_eq_viz"] = np.nan
        feats["mrz_sex_eq_viz"]   = np.nan

        feats["mrz_number_tail_eq"]    = np.nan
        feats["mrz_number_head_eq"]    = np.nan
        feats["mrz_number_full_eq"]    = np.nan
        feats["mrz_number_is_subseq"]  = np.nan
        feats["mrz_code_in_personal"]  = np.nan

        feats["mrz_surname_eq"]          = np.nan
        feats["mrz_surname_initial_eq"]  = np.nan
        feats["mrz_surname_charsim"]     = np.nan

        feats["mrz_given_n_parts"]      = 0.0   # факт: неизвестно
        feats["mrz_given_structure_ok"] = np.nan
        feats["mrz_firstname_charsim"]  = np.nan
        feats["mrz_middlename_charsim"] = np.nan

        feats["mrz_agreement_sum"]  = np.nan
        feats["mrz_agreement_min"]  = np.nan
        feats["mrz_agreement_mean"] = np.nan

        feats["name_sim_min"]  = np.nan
        feats["name_sim_mean"] = np.nan
        feats["name_sim_low"]  = np.nan


    feats["viz_given_n_parts"] = float(len(viz_given_parts))

    
    # ============ справочник подразделений ============
    code = corrected.get("Issue_organisation_code", "")
    org = corrected.get("Issue_organization_ru", "")
    ref_names = subdivisions.get(code) if subdivisions else None

    if ref_names:
        feats["subdiv_code_in_ref"] = 1.0
        feats["subdiv_n_names"] = float(len(ref_names))
        ocr_geo = _geo_tokens(org)
        ocr_up = org.upper()
        best_fwd = best_rev = 0.0
        for nm in ref_names:
            ref_geo = _geo_tokens(nm)
            if ocr_geo and ref_geo:
                best_fwd = max(best_fwd, len(ocr_geo & ref_geo) / len(ocr_geo))
            if ref_geo:
                best_rev = max(best_rev,
                               sum(1 for t in ref_geo if t in ocr_up) / len(ref_geo))
        feats["subdiv_geo_hit"] = best_fwd
        feats["subdiv_ref_hit"] = best_rev
    else:
        feats["subdiv_code_in_ref"] = 0.0
        feats["subdiv_n_names"]     = 0.0
        feats["subdiv_geo_hit"]     = np.nan
        feats["subdiv_ref_hit"]     = np.nan

    return feats, corrected

