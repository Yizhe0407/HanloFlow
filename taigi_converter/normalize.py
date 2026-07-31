import re
import unicodedata
from datetime import date

from .unicode_policy import HAN_IDEOGRAPH_CLASS

_CJK_CHAR_CLASS = HAN_IDEOGRAPH_CLASS
_CJK_PUNCT_CLASS = "，。！？；：、"
_DIGITS = "零一二三四五六七八九"
_BIG_UNITS = ["", "萬", "億", "兆", "京"]
_MAX_UNIT_DIGITS = len(_BIG_UNITS) * 4
_FULLWIDTH_DIGIT_TRANS = str.maketrans("０１２３４５６７８９", "0123456789")
_S2T_TRANS = str.maketrans(
    {
        "这": "這",
        "车": "車",
        "门": "門",
        "们": "們",
        # 「里／裡」在繁體中文是不同字（公里、鄉里、裡面），不可做
        # 無條件單字轉換。產品輸入契約是繁體中文，normalizer 應保守地
        # 保留合法繁體詞形，而不是猜測簡體語境。
        "线": "線",
        "发": "發",
        "听": "聽",
        "说": "說",
        "点": "點",
        "时": "時",
        "问": "問",
        "会": "會",
        "过": "過",
        # 「后／後」同樣是繁體異義字（太后、皇后、後來），不可做
        # 無條件單字轉換。
        "让": "讓",
        "吗": "嗎",
        "云": "雲",
    }
)
_VARIANT_TRANS = str.maketrans(
    {
        "喺": "在",
        "睇": "看",
        "咗": "了",
        "啲": "的",
        "啱": "著",
        "嚟": "來",
        "攞": "拿",
        "嗰": "那",
    }
)
_NUMBER_TOKEN_RE = re.compile(r"(?<![A-Za-z0-9_.])([+-]?\d+(?:\.\d+)?)(?![A-Za-z0-9_.])")

# 數字正規化前先辨識完整 technical span。這些模式刻意要求明確的語法
# 訊號（scheme、路徑分隔符、比較運算子、固定 identifier 格式等），避免把
# 一般金額、負數、日期簡寫或數量範圍誤判成技術資料。
_IPV4_OCTET = r"(?:25[0-5]|2[0-4][0-9]|1[0-9]{2}|[1-9]?[0-9])"
_PORT = r"(?:0|[1-9][0-9]{0,3}|[1-5][0-9]{4}|6[0-4][0-9]{3}|65[0-4][0-9]{2}|655[0-2][0-9]|6553[0-5])"
_VERSION = r"[vV]?[0-9](?:[A-Za-z0-9.*+!_-]|\.(?=[A-Za-z0-9]))*"
_VERSION_COMPARATOR = rf"(?:===|==|~=|!=|<=|>=|<|>)\s*{_VERSION}"
_PACKAGE_NAME = r"(?:@?[A-Za-z][A-Za-z0-9_.-]*(?:/[A-Za-z][A-Za-z0-9_.-]*)?)"

# Only domain-specific labels that unambiguously introduce an identifier belong
# here.  Keep generic ``編號`` out: in ordinary prose its following digits still
# use the existing digit-by-digit normalization contract.  Extending support is
# intentionally a data-only change to this allowlist plus tests.
_IDENTIFIER_LABELS = (
    "案件編號",
    "訂單編號",
    "案號",
    "THSR",
)
_IDENTIFIER_LABEL_PATTERN = "|".join(re.escape(label) for label in sorted(_IDENTIFIER_LABELS, key=len, reverse=True))
_HORIZONTAL_SPACE = r"[ \t\u3000\xA0]"
_IDENTIFIER_VALUE_SEPARATOR = rf"(?:{_HORIZONTAL_SPACE}+|{_HORIZONTAL_SPACE}*(?:是|為|[:：=]){_HORIZONTAL_SPACE}*)"
_IDENTIFIER_SPAN_RE = re.compile(
    rf"(?:{_IDENTIFIER_LABEL_PATTERN}){_IDENTIFIER_VALUE_SEPARATOR}[0-9]+(?![A-Za-z0-9_.+/-])"
)

# A spaced, unit-qualified calendar date is an explicit structured value in
# forms, schedules, and machine-readable messages. Requiring horizontal space
# at every boundary keeps the existing compact-news contract intact: ``2026年``
# and ``6/1`` remain ordinary prose numbers. The regex only identifies
# candidates; ``_valid_formal_date_spans`` rejects impossible Gregorian dates.
_FORMAL_DATE_SPAN_RE = re.compile(
    rf"(?<![0-9])(?P<year>[0-9]{{4}}){_HORIZONTAL_SPACE}+年"
    rf"{_HORIZONTAL_SPACE}+(?P<month>0?[1-9]|1[0-2]){_HORIZONTAL_SPACE}+月"
    rf"{_HORIZONTAL_SPACE}+(?P<day>0?[1-9]|[12][0-9]|3[01]){_HORIZONTAL_SPACE}+日(?![0-9])"
)

_TECHNICAL_SPAN_RES = (
    _IDENTIFIER_SPAN_RE,
    # Decimal percentages, clock times, and room numbers are structured values/identifiers.
    # Preserve their ASCII surface form so decimals cannot be reordered into invalid
    # forms such as ``一點百分之七五`` and downstream consumers retain exact tokens.
    # Whole-number percentages keep the established ``百分之N`` speech contract.
    re.compile(r"(?<![A-Za-z0-9_.])[+-]?[0-9]+\.[0-9]+%(?![A-Za-z0-9_.])"),
    re.compile(r"(?<![0-9])(?:[01][0-9]|2[0-3]):[0-5][0-9](?![0-9])"),
    re.compile(r"(?<![A-Za-z0-9])(?:[1-9][0-9]{0,5})[ \t\u3000\xA0]*室(?![A-Za-z0-9])"),
    # Hash-prefixed issue/ticket identifiers are opaque references, not prose numbers.
    # Preserve both standalone forms (``#431``) and conventional ASCII labels
    # (``PR#908``, ``GH#123``) as one technical span.
    re.compile(r"(?<![A-Za-z0-9_])[A-Za-z][A-Za-z0-9_.-]*#[0-9]+(?![A-Za-z0-9_])"),
    re.compile(r"(?<![A-Za-z0-9_])#[0-9]+(?![A-Za-z0-9_])"),
    re.compile(r"\b(?:https?|ftp)://[^\s<>\"'，。！？；]+", re.IGNORECASE),
    re.compile(rf"(?<![A-Za-z0-9_.])(?:{_IPV4_OCTET}\.){{3}}{_IPV4_OCTET}(?::{_PORT})?(?![A-Za-z0-9_.])"),
    re.compile(r"(?<![A-Za-z0-9])CVE-[0-9]{4}-[0-9]{4,}(?![A-Za-z0-9])", re.IGNORECASE),
    re.compile(
        r"(?<![A-Fa-f0-9])[A-Fa-f0-9]{8}-[A-Fa-f0-9]{4}-[1-5][A-Fa-f0-9]{3}-"
        r"[89ABab][A-Fa-f0-9]{3}-[A-Fa-f0-9]{12}(?![A-Fa-f0-9])"
    ),
    re.compile(r"(?<![A-Za-z0-9])/(?:[A-Za-z0-9._~+%-]+/)*[A-Za-z0-9._~+%-]+"),
    re.compile(r"(?<![A-Za-z0-9])[A-Za-z]:\\(?:[^\s\\/:*?\"<>|]+\\)*[^\s\\/:*?\"<>|]+"),
    re.compile(r"(?<![A-Za-z0-9])\\\\[^\s\\/:*?\"<>|]+\\[^\s\\/:*?\"<>|]+(?:\\[^\s\\/:*?\"<>|]+)*"),
    re.compile(
        r"(?<![A-Za-z0-9_])[?&][A-Za-z_][A-Za-z0-9_.-]*=[^&\s#，。！？；]+"
        r"(?:&[A-Za-z_][A-Za-z0-9_.-]*=[^&\s#，。！？；]+)*"
    ),
    re.compile(
        rf"(?<![A-Za-z0-9_@./-]){_PACKAGE_NAME}(?:\[[A-Za-z0-9_,.-]+\])?\s*"
        rf"{_VERSION_COMPARATOR}(?:\s*,\s*{_VERSION_COMPARATOR})*(?![A-Za-z0-9_.-])"
    ),
    re.compile(rf"(?<![A-Za-z0-9_@./-]){_PACKAGE_NAME}@{_VERSION}(?![A-Za-z0-9_.-])"),
    re.compile(
        r"(?<![A-Za-z0-9_./-])(?=[a-z0-9_./-]*[a-z])"
        r"(?:[a-z0-9]+(?:[._-][a-z0-9]+)*/)*"
        r"[a-z0-9]+(?:[._-][a-z0-9]+)*:[vV]?[0-9][A-Za-z0-9_.-]*(?![A-Za-z0-9_.-])"
    ),
    re.compile(
        r"(?<![A-Za-z0-9])(?:[0-9]{4}-[0-9]{2}-[0-9]{2}|"
        r"(?=[A-Za-z0-9-]*[A-Za-z])[A-Za-z0-9]+(?:-[A-Za-z0-9]+)+)(?![A-Za-z0-9])"
    ),
)
_TECHNICAL_NUMBER_PREFIX_RE = re.compile(
    r"(?:HTTP(?:/[0-9](?:\.[0-9])?)?|HTTPS|Python|Node(?:\.js)?|TLS|SSL|RFC|ISO|UTF-?8|Unicode)\s+$",
    re.IGNORECASE,
)


def _int_to_han(n: int) -> str:
    if n == 0:
        return _DIGITS[0]

    def group_to_han(group: int) -> str:
        units = ["", "十", "百", "千"]
        out: list[str] = []
        pending_zero = False
        for i in range(3, -1, -1):
            base = 10**i
            digit = (group // base) % 10
            if digit == 0:
                if out and group % base:
                    pending_zero = True
                continue
            if pending_zero:
                out.append("零")
                pending_zero = False
            if i == 1 and digit == 1 and not out:
                out.append("十")
            else:
                out.append(_DIGITS[digit] + units[i])
        return "".join(out)

    groups: list[int] = []
    value = n
    while value:
        groups.append(value % 10000)
        value //= 10000

    if len(groups) > len(_BIG_UNITS):
        return "".join(_DIGITS[int(ch)] for ch in str(n))

    out: list[str] = []
    pending_zero = False
    for idx in range(len(groups) - 1, -1, -1):
        group = groups[idx]
        if group == 0:
            if out and any(g != 0 for g in groups[:idx]):
                pending_zero = True
            continue
        if pending_zero:
            out.append("零")
            pending_zero = False
        out.append(group_to_han(group) + _BIG_UNITS[idx])
        if idx > 0 and 0 < groups[idx - 1] < 1000:
            pending_zero = True
    return "".join(out)


def _ascii_decimal_digits(digits: str) -> str:
    # ``\d`` 也會命中 Unicode 十進位數字；先轉成 ASCII，避免不同字形
    # 在後續轉換中有不一致的行為。
    return "".join(str(unicodedata.decimal(ch)) for ch in digits)


def _digits_to_han(digits: str) -> str:
    return "".join(_DIGITS[ord(ch) - ord("0")] for ch in digits)


def _integer_digits_to_han(digits: str) -> str:
    # 前導零常用於編號、版本或精確輸入，應逐位保留，不能先轉 int。
    if len(digits) > 1 and digits.startswith("0"):
        return _digits_to_han(digits)

    # 現有單位只到「京」。更長數字原本也是逐位輸出；在 int() 前先
    # fallback，避免觸發 Python 的 int_max_str_digits 安全限制。
    if len(digits) > _MAX_UNIT_DIGITS:
        return _digits_to_han(digits)

    return _int_to_han(int(digits))


def _technical_spans(text: str) -> tuple[tuple[int, int], ...]:
    """Return merged ranges whose ASCII digits must remain byte-for-byte stable."""

    spans = [match.span() for pattern in _TECHNICAL_SPAN_RES for match in pattern.finditer(text)]
    spans.extend(_valid_formal_date_spans(text))
    spans.sort()
    if not spans:
        return ()

    merged: list[tuple[int, int]] = []
    for start, end in spans:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return tuple(merged)


def _valid_formal_date_spans(text: str) -> tuple[tuple[int, int], ...]:
    """Return explicit spaced calendar dates that are valid Gregorian dates."""

    spans: list[tuple[int, int]] = []
    for match in _FORMAL_DATE_SPAN_RE.finditer(text):
        try:
            date(
                int(match.group("year")),
                int(match.group("month")),
                int(match.group("day")),
            )
        except ValueError:
            continue
        spans.append(match.span())
    return tuple(spans)


def _number_is_structured_literal(
    text: str,
    start: int,
    end: int,
    technical_spans: tuple[tuple[int, int], ...],
) -> bool:
    """Return whether a numeric token belongs to an already identified technical span."""

    if any(span_start <= start and end <= span_end for span_start, span_end in technical_spans):
        return True

    # Whitespace-delimited runtime/protocol forms (for example ``Python 3.11``
    # and ``HTTP 429``) are inherently ambiguous without a delimiter. Keep the
    # small compatibility set here; all structured forms above are syntax-driven
    # and do not require adding product names.
    prefix = text[max(0, start - 32) : start]
    return _TECHNICAL_NUMBER_PREFIX_RE.search(prefix) is not None


def _replace_number_token(
    text: str,
    match: re.Match[str],
    technical_spans: tuple[tuple[int, int], ...],
) -> str:
    start, end = match.span(1)
    token = match.group(1)
    if _number_is_structured_literal(text, start, end, technical_spans):
        return token
    return _number_to_han(token)


def _number_to_han(token: str) -> str:
    sign = ""
    body = token
    if body.startswith("+"):
        body = body[1:]
    elif body.startswith("-"):
        sign = "負"
        body = body[1:]

    if "." in body:
        int_part, frac_part = body.split(".", 1)
        int_digits = _ascii_decimal_digits(int_part)
        frac_digits = _ascii_decimal_digits(frac_part)
        return f"{sign}{_integer_digits_to_han(int_digits)}點{_digits_to_han(frac_digits)}"

    digits = _ascii_decimal_digits(body)
    return sign + _integer_digits_to_han(digits)


def normalize_cjk_spacing(
    text: str,
    *,
    trim_outer: bool = False,
) -> str:
    """Canonicalize horizontal spacing without changing glyphs or numbers."""

    if trim_outer:
        text = text.strip()
    # 只壓縮空白/Tab/全形空白/NBSP，保留換行供段落與 TTS 斷句使用。
    text = re.sub(r"[ \t\u3000\xA0]+", " ", text)
    # 中文標點前後只移除水平空白，避免吞掉換行。
    text = re.sub(
        rf"[ \t\u3000\xA0]*([{_CJK_PUNCT_CLASS}])[ \t\u3000\xA0]*",
        r"\1",
        text,
    )
    # 兩側皆為 CJK 時移除水平空白。這一步也供 converter 在規則新增
    # CJK 邊界後做最終 canonicalization，避免第二輪才消失的空白。
    return re.sub(
        rf"(?<=[{_CJK_CHAR_CLASS}])[ \t\u3000\xA0]+(?=[{_CJK_CHAR_CLASS}])",
        "",
        text,
    )


def normalize_text(
    text: str,
    *,
    compress_spaces: bool = True,
    trim_outer: bool = True,
    convert_numbers: bool = True,
) -> str:
    if trim_outer:
        text = text.strip()

    # 統一常見字形
    text = text.replace("臺", "台")
    text = text.translate(_S2T_TRANS)
    text = text.translate(_VARIANT_TRANS)
    text = text.translate(_FULLWIDTH_DIGIT_TRANS)

    # 先辨識完整 technical spans，再將其餘獨立數字 token 轉為漢字。若先
    # 處理 CJK 空白，「一週 5人」首輪會保留空白，第二輪才因「5」已成
    # 「五」而移除，造成 normalize_text 本身不具冪等性。Converter 會先以
    # ``convert_numbers=False`` 做字形正規化與詞典保護，再只對未保護片段
    # 執行數字轉換，讓含數字的專名可由同一套 protected-term 治理。
    if convert_numbers:
        technical_spans = _technical_spans(text)
        text = _NUMBER_TOKEN_RE.sub(
            lambda match: _replace_number_token(text, match, technical_spans),
            text,
        )

    if compress_spaces:
        text = normalize_cjk_spacing(text)

    return text
