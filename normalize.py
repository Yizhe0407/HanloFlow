import re


_CJK_CHAR_CLASS = r"\u3400-\u9fff\uf900-\ufaff\U00020000-\U0002FA1F"
_CJK_PUNCT_CLASS = "，。！？；：、"
_DIGITS = "零一二三四五六七八九"
_BIG_UNITS = ["", "萬", "億", "兆", "京"]
_FULLWIDTH_DIGIT_TRANS = str.maketrans("０１２３４５６７８９", "0123456789")
_NUMBER_TOKEN_RE = re.compile(r"(?<![A-Za-z0-9_.])([+-]?\d+(?:\.\d+)?)(?![A-Za-z0-9_.])")


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
        int_han = _int_to_han(int(int_part))
        frac_han = "".join(_DIGITS[int(ch)] for ch in frac_part)
        return f"{sign}{int_han}點{frac_han}"

    if len(body) > 1 and body.startswith("0"):
        return sign + "".join(_DIGITS[int(ch)] for ch in body)
    return sign + _int_to_han(int(body))


def normalize_text(
    text: str,
    *,
    compress_spaces: bool = True,
    trim_outer: bool = True,
) -> str:
    if trim_outer:
        text = text.strip()

    # 統一常見字形
    text = text.replace("臺", "台")
    text = text.translate(_FULLWIDTH_DIGIT_TRANS)

    if compress_spaces:
        # 只壓縮空白/Tab/全形空白/NBSP，保留換行供段落與 TTS 斷句使用。
        text = re.sub(r"[ \t\u3000\xA0]+", " ", text)

        # 中文標點前後只移除空白/Tab/全形空白/NBSP，避免吞掉換行。
        text = re.sub(rf"[ \t\u3000\xA0]*([{_CJK_PUNCT_CLASS}])[ \t\u3000\xA0]*", r"\1", text)

        # 兩側皆為 CJK 時只移除空白/Tab/全形空白/NBSP，不跨行壓縮。
        text = re.sub(rf"(?<=[{_CJK_CHAR_CLASS}])[ \t\u3000\xA0]+(?=[{_CJK_CHAR_CLASS}])", "", text)

    # 將獨立數字 token 轉為漢字，避免 A12 這類英數代碼被誤改。
    text = _NUMBER_TOKEN_RE.sub(lambda m: _number_to_han(m.group(1)), text)

    return text
