from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.regression_runner import (
    RegressionCase as RegressionCaseModel,
)
from scripts.regression_runner import (
    compatibility_snapshot_case as RegressionCase,
)
from scripts.regression_runner import (
    run_regression_cli,
)

TAXI_REGRESSION_CASES: list[RegressionCaseModel] = [
    # hailing — 叫車
    RegressionCase("hailing", "我要叫計程車。", "我欲叫計程車。"),
    RegressionCase("hailing", "請問有沒有叫車服務？", "借問敢有叫車服務？"),
    RegressionCase("hailing", "司機快來了嗎？", "司機緊來矣無？"),
    RegressionCase("hailing", "可以幫我叫一台計程車嗎？", "會當替我叫一台計程車無？"),
    RegressionCase("hailing", "我在飯店門口等車。", "我佇飯店門跤口等車。"),
    RegressionCase("hailing", "請在門口等我。", "請佇門跤口等我。"),
    RegressionCase("hailing", "我們有四個人。", "阮有四个人。"),
    RegressionCase("hailing", "可以派大一點的車嗎？", "會當派較大的車無？"),
    RegressionCase("hailing", "車可以等五分鐘嗎？", "車會當等五分鐘無？"),
    RegressionCase("hailing", "我在便利商店門口等車。", "我佇便利店門跤口等車。"),
    RegressionCase("hailing", "我想晚一點出發。", "我想欲較晏一點出發。"),
    RegressionCase("hailing", "可以幫我叫大車嗎？", "會當替我叫大車無？"),
    RegressionCase("hailing", "可以幫我改叫車時間嗎？", "會當替我改叫車時間無？"),
    RegressionCase("hailing", "可以幫我查派車進度嗎？", "會當替我查派車進度無？"),
    RegressionCase("hailing", "我想改出發時間。", "我想欲改出發時間。"),
    RegressionCase("hailing", "可以幫我查叫車紀錄嗎？", "會當替我查叫車紀錄無？"),
    RegressionCase("hailing", "可以幫我查乘車紀錄嗎？", "會當替我查乘車紀錄無？"),
    # destination — 目的地
    RegressionCase("destination", "請問到台北車站多少錢？", "借問到臺北車站偌濟錢？"),
    RegressionCase("destination", "我要去機場。", "我欲去機場。"),
    RegressionCase("destination", "請到這個地址。", "請到這个地址。"),
    RegressionCase("destination", "請送我到機場。", "請載我到機場。"),
    RegressionCase("destination", "我要去高鐵站。", "我欲去高鐵站。"),
    RegressionCase("destination", "我想改目的地。", "我想欲改目的地。"),
    RegressionCase("destination", "請走高速公路。", "請走高速公路。"),
    RegressionCase("destination", "我要到捷運站。", "我欲到捷運站。"),
    RegressionCase("destination", "我要到醫院。", "我欲到病院。"),
    RegressionCase("destination", "請載我到飯店。", "請載我到飯店。"),
    RegressionCase("destination", "到機場要多久？", "到機場愛偌久？"),
    RegressionCase("destination", "我想先去加油站。", "我想欲先去加油站。"),
    RegressionCase("destination", "我想先去便利商店。", "我想欲先去便利店。"),
    RegressionCase("destination", "可以幫我改上車地點嗎？", "會當替我改上車所在無？"),
    RegressionCase("destination", "我想改上車時間。", "我想欲改上車時間。"),
    RegressionCase("destination", "我想改下車地點。", "我想欲改落車地點。"),
    RegressionCase("destination", "可以幫我改目的地嗎？", "會當替我改目的地無？"),
    RegressionCase("destination", "可以幫我改下車地址嗎？", "會當替我改落車地址無？"),
    # navigation — 行進指引
    RegressionCase("navigation", "請停在前面。", "請停佇頭前。"),
    RegressionCase("navigation", "在前面右轉。", "佇頭前正斡。"),
    RegressionCase("navigation", "在前面左轉。", "佇頭前倒斡。"),
    RegressionCase("navigation", "直走就到了。", "直走就到矣。"),
    RegressionCase("navigation", "就這裡下車。", "就遮落車。"),
    RegressionCase("navigation", "麻煩靠邊停一下。", "麻煩靠路爿停一下。"),
    RegressionCase("navigation", "前面臨停一下就好。", "頭前暫停一下就好。"),
    RegressionCase("navigation", "靠右邊停。", "靠正手爿停。"),
    RegressionCase("navigation", "靠左邊停。", "靠倒手爿停。"),
    RegressionCase("navigation", "前面路口右轉。", "頭前路口正斡。"),
    RegressionCase("navigation", "不要走高速公路。", "莫走高速公路。"),
    RegressionCase("navigation", "可以迴轉嗎？", "會當踅頭無？"),
    # payment — 付款
    RegressionCase("payment", "多少錢？", "偌濟錢？"),
    RegressionCase("payment", "不用找了。", "免找矣。"),
    RegressionCase(
        "payment",
        "可以刷卡嗎？",
        "會當刷卡無？",
        duplicate_group="cross_domain_payment_intent",
        duplicate_reason="餐飲、購物與計程車產品面共用付款意圖，但需各自保留端到端覆蓋。",
    ),
    RegressionCase("payment", "可以開收據嗎？", "會當開收據無？"),
    RegressionCase(
        "payment",
        "可以用電子支付嗎？",
        "會當用電子支付無？",
        duplicate_group="cross_domain_payment_intent",
        duplicate_reason="餐飲、購物與計程車產品面共用付款意圖，但需各自保留端到端覆蓋。",
    ),
    RegressionCase(
        "payment",
        "可以開發票嗎？",
        "會當開發票無？",
        duplicate_group="cross_domain_payment_intent",
        duplicate_reason="餐飲、購物與計程車產品面共用付款意圖，但需各自保留端到端覆蓋。",
    ),
    RegressionCase("payment", "不用找零了。", "免找錢矣。"),
    RegressionCase("payment", "我想改成現金付款。", "我想欲改做現錢付款。"),
    RegressionCase("payment", "可以幫我查車資嗎？", "會當替我查車錢無？"),
    RegressionCase("payment", "可以幫我查預估車資嗎？", "會當替我查預估車錢無？"),
    RegressionCase("payment", "可以幫我查共乘車資嗎？", "會當替我查共乘車錢無？"),
    RegressionCase("payment", "可以幫我查車資明細嗎？", "會當替我查車錢明細無？"),
    # misc — 其他
    RegressionCase("misc", "請快一點。", "請較緊咧。"),
    RegressionCase("misc", "等我一下。", "等我一下仔。"),
    RegressionCase("misc", "到了叫我。", "到矣叫我。"),
    RegressionCase("misc", "打開後車廂。", "拍開後行李箱。"),
    RegressionCase("misc", "車牌號碼是多少？", "車牌號碼是幾號？"),
    RegressionCase("misc", "我東西忘在車上了。", "我的物件放袂記佇車頂矣。"),
    RegressionCase("misc", "可以聯絡司機嗎？", "會當聯絡司機無？"),
    RegressionCase("misc", "我在前面下車。", "我佇頭前落車。"),
    RegressionCase("misc", "可以幫我開後車廂嗎？", "會當替我開後行李箱無？"),
    RegressionCase("misc", "請幫我等一下。", "請替我等一下仔。"),
    RegressionCase("misc", "可以幫我聯絡司機嗎？", "會當替我聯絡司機無？"),
    RegressionCase("misc", "可以幫我查司機電話嗎？", "會當替我查司機電話無？"),
    RegressionCase("misc", "可以幫我查車牌號碼嗎？", "會當替我查車牌號碼無？"),
    RegressionCase("misc", "可以幫我查車子位置嗎？", "會當替我查車子位置無？"),
    RegressionCase("misc", "可以幫我查司機位置嗎？", "會當替我查司機位置無？"),
    RegressionCase("misc", "可以幫我查司機姓名嗎？", "會當替我查司機姓名無？"),
    RegressionCase("misc", "可以幫我看一下路線嗎？", "會當替我看覓路線無？"),
    RegressionCase("misc", "請幫我開窗戶。", "請替我開窗仔門。"),
    RegressionCase("misc", "可以幫我查車型嗎？", "會當替我查車型無？"),
    RegressionCase("misc", "可以幫我查車輛位置嗎？", "會當替我查車輛位置無？"),
    RegressionCase("misc", "可以幫我查司機評價嗎？", "會當替我查司機評價無？"),
    RegressionCase("misc", "可以幫乘客確認上車地點嗎？", "會當替乘客確認上車所在無？"),
    RegressionCase("misc", "可不可以幫乘客改下車地點？", "敢會當替乘客改落車地點？"),
    RegressionCase("misc", "請協助乘客上車。", "請鬥相共乘客上車。"),
]


def main() -> int:
    return run_regression_cli(
        TAXI_REGRESSION_CASES,
        description="計程車情境 regression runner",
    )


if __name__ == "__main__":
    raise SystemExit(main())
