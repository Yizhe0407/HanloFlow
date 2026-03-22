import re


def normalize_text(text: str) -> str:
    text = text.strip()

    # 統一常見字形
    text = text.replace("臺", "台")

    # 清多餘空白
    text = re.sub(r"\s+", "", text)

    return text
