from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.regression_runner import RegressionCase, run_regression_cli

FAMILY_REGRESSION_CASES: list[RegressionCase] = [
    # parent_child — 親子日常
    RegressionCase("parent_child", "媽媽在煮飯。", "阿母佇咧煮飯。"),
    RegressionCase("parent_child", "爸爸去上班了。", "阿爸去上班矣。"),
    RegressionCase("parent_child", "孩子在哭。", "囡仔佇咧哭。"),
    RegressionCase("parent_child", "小孩在玩玩具。", "囡仔佇咧玩𨑨迌物仔。"),
    RegressionCase("parent_child", "孩子該起床了。", "囡仔愛起床矣。"),
    RegressionCase("parent_child", "下午去接孩子。", "下晝去接囡仔。"),
    RegressionCase("parent_child", "幫孩子換衣服。", "幫囡仔換衫。"),
    RegressionCase("parent_child", "不要哭了。", "毋通哭矣。"),
    RegressionCase("parent_child", "我帶小孩去學校。", "我帶囡仔去學校。"),
    RegressionCase("parent_child", "可以幫我接小孩嗎？", "會當替我接囡仔無？"),
    RegressionCase("parent_child", "我晚點帶小孩回家。", "我較晏帶囡仔轉去厝裡。"),
    RegressionCase("parent_child", "我晚點去接小孩。", "我較晏去接囡仔。"),
    RegressionCase("parent_child", "我想改接小孩時間。", "我想欲改接囡仔時間。"),
    RegressionCase("parent_child", "請協助改接小孩時間。", "請鬥相共改接囡仔時間。"),
    RegressionCase("parent_child", "可以幫我查保母電話嗎？", "會當替我查保母電話無？"),
    RegressionCase("parent_child", "可以幫我查孩子作業嗎？", "會當替我查囡仔作業無？"),
    RegressionCase("parent_child", "可以幫我查孩子成績嗎？", "會當替我查囡仔成績無？"),
    RegressionCase("parent_child", "請協助查孩子成績。", "請鬥相共查囡仔成績。"),
    RegressionCase("parent_child", "我需要確認孩子成績。", "我欲確認囡仔成績。"),
    RegressionCase("parent_child", "我想確認孩子成績。", "我想欲確認囡仔成績。"),
    RegressionCase("parent_child", "可以幫我確認孩子成績嗎？", "會當替我確認囡仔成績無？"),
    RegressionCase("parent_child", "請幫我確認孩子成績。", "請替我確認囡仔成績。"),
    RegressionCase("parent_child", "麻煩你幫我確認孩子成績。", "麻煩你替我確認囡仔成績。"),
    RegressionCase("parent_child", "幫我確認孩子成績。", "替我確認囡仔成績。"),
    RegressionCase("parent_child", "請協助確認孩子成績。", "請鬥相共確認囡仔成績。"),
    # health_care — 照護
    RegressionCase("health_care", "孩子發燒了。", "囡仔發燒矣。"),
    RegressionCase("health_care", "幫孩子洗澡。", "幫囡仔洗身軀。"),
    RegressionCase("health_care", "帶孩子去看醫生。", "帶囡仔去予醫生看。"),
    RegressionCase("health_care", "孩子在睡午覺。", "囡仔佇咧睏晝。"),
    RegressionCase("health_care", "孩子要吃藥。", "囡仔愛食藥仔。"),
    RegressionCase("health_care", "藥要按時吃。", "藥仔愛照時間食。"),
    RegressionCase("health_care", "媽媽要休息。", "阿母欲歇睏。"),
    RegressionCase("health_care", "我陪你去看醫生。", "我陪你去予醫生看。"),
    RegressionCase("health_care", "媽媽說晚上要吃藥。", "阿母講暗時愛食藥仔。"),
    RegressionCase("health_care", "爸爸明天要回診。", "阿爸明仔載愛回診。"),
    RegressionCase("health_care", "今天要帶小孩去看醫生。", "今仔日愛帶囡仔去予醫生看。"),
    RegressionCase("health_care", "小孩肚子痛。", "囡仔腹肚疼。"),
    RegressionCase("health_care", "可以幫我買藥嗎？", "會當替我買藥仔無？"),
    RegressionCase("health_care", "可以幫我查藥袋嗎？", "會當替我查藥袋無？"),
    RegressionCase("health_care", "可以幫我查小孩體溫嗎？", "會當替我查囡仔體溫無？"),
    RegressionCase("health_care", "可以幫我查奶粉庫存嗎？", "會當替我查奶粉庫存無？"),
    RegressionCase("health_care", "可以幫我查尿布庫存嗎？", "會當替我查尿苴仔庫存無？"),
    RegressionCase("health_care", "可以幫我查小孩作息嗎？", "會當替我查囡仔作息無？"),
    RegressionCase("health_care", "可以幫我查孩子疫苗嗎？", "會當替我查囡仔疫苗無？"),
    RegressionCase("health_care", "可以幫我查小孩過敏紀錄嗎？", "會當替我查囡仔過敏紀錄無？"),
    RegressionCase("health_care", "可以幫我查孩子出勤嗎？", "會當替我查囡仔出勤無？"),
    RegressionCase("health_care", "可以幫我查孩子聯絡簿嗎？", "會當替我查囡仔聯絡簿無？"),
    RegressionCase("health_care", "可以幫我查孩子疫苗紀錄嗎？", "會當替我查囡仔疫苗紀錄無？"),
    # siblings — 兄弟姐妹
    RegressionCase("siblings", "姐姐在讀書。", "阿姊佇咧讀冊。"),
    RegressionCase("siblings", "弟弟在做功課。", "阿弟仔佇咧做功課。"),
    RegressionCase("siblings", "妹妹在睡午覺。", "小妹仔佇咧睏晝。"),
    RegressionCase("siblings", "哥哥去上學了。", "阿兄去上課矣。"),
    RegressionCase("siblings", "哥哥妹妹在吵架。", "阿兄小妹仔佇咧冤家。"),
    RegressionCase("siblings", "姐姐幫弟弟拿書包。", "阿姊幫阿弟仔提冊包。"),
    RegressionCase("siblings", "兄弟姐妹要一起分享玩具。", "兄弟姊妹愛鬥陣分享𨑨迌物仔。"),
    RegressionCase("siblings", "妹妹不想寫功課。", "小妹仔毋想欲寫功課。"),
    RegressionCase("siblings", "請幫我照顧弟弟。", "請替我照顧阿弟仔。"),
    # grandparents — 祖父母
    RegressionCase("grandparents", "爺爺在下棋。", "阿公佇咧行棋。"),
    RegressionCase("grandparents", "奶奶在洗碗。", "阿媽佇咧洗碗。"),
    RegressionCase("grandparents", "帶孩子去找爺爺奶奶。", "帶囡仔去找阿公阿媽。"),
    RegressionCase("grandparents", "爺爺在看電視。", "阿公佇咧看電視。"),
    RegressionCase("grandparents", "明天去看爺爺奶奶。", "明仔載去看阿公阿媽。"),
    RegressionCase("grandparents", "阿公身體還好嗎？", "阿公身體敢猶好？"),
    RegressionCase("grandparents", "奶奶要去散步。", "阿媽欲去散步。"),
    RegressionCase("grandparents", "打電話給爺爺。", "拍電話予阿公。"),
    # daily — 家庭日常
    RegressionCase("daily", "老婆在洗衣服。", "某佇咧洗衫。"),
    RegressionCase("daily", "老公在工作。", "翁婿佇咧工作。"),
    RegressionCase("daily", "女兒很乖。", "查某囝真乖。"),
    RegressionCase("daily", "爸爸在喝茶。", "阿爸佇咧啉茶。"),
    RegressionCase("daily", "爸爸要出門了。", "阿爸欲出門矣。"),
    RegressionCase("daily", "晚餐好了。", "暗頓好矣。"),
    RegressionCase("daily", "我們一起吃飯。", "咱鬥陣食飯。"),
    RegressionCase("daily", "記得洗碗。", "記著洗碗。"),
    RegressionCase("daily", "我可以晚一點回家嗎？", "我會當較晏一點轉去厝裡無？"),
    RegressionCase("daily", "請幫我買晚餐。", "請替我買暗頓。"),
    RegressionCase("daily", "晚餐要吃什麼？", "暗頓欲食啥？"),
    RegressionCase("daily", "晚餐吃什麼？", "暗頓食啥？"),
    RegressionCase("daily", "可以幫我買早餐嗎？", "會當替我買早頓無？"),
    RegressionCase("daily", "可以幫我煮晚餐嗎？", "會當替我煮暗頓無？"),
    RegressionCase("daily", "我想改晚餐時間。", "我想欲改暗頓時間。"),
    RegressionCase("daily", "可以幫我查垃圾車時間嗎？", "會當替我查糞埽車時間無？"),
    RegressionCase("daily", "可以幫我倒垃圾嗎？", "會當替我摒糞埽無？"),
    RegressionCase("daily", "我想改午餐時間。", "我想欲改晝頓時間。"),
    RegressionCase("daily", "可以幫我提醒媽媽嗎？", "會當替我提醒阿母無？"),
    RegressionCase("daily", "我想改晚睡時間。", "我想欲改晚睡時間。"),
    RegressionCase("daily", "可以幫我提醒爸爸嗎？", "會當替我提醒阿爸無？"),
    RegressionCase("daily", "我想改回家時間。", "我想欲改轉去厝裡時間。"),
    RegressionCase("daily", "我想改睡覺時間。", "我想欲改睏眠時間。"),
]


def main() -> int:
    return run_regression_cli(
        FAMILY_REGRESSION_CASES,
        description='家庭/親子情境 regression runner',
    )


if __name__ == "__main__":
    raise SystemExit(main())
