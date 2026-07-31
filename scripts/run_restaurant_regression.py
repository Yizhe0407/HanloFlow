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

RESTAURANT_REGRESSION_CASES: list[RegressionCaseModel] = [
    # ordering — 點餐
    RegressionCase("ordering", "我要點餐。", "我欲點餐。"),
    RegressionCase("ordering", "請給我菜單。", "請予我菜單。"),
    RegressionCase("ordering", "請問今天有什麼特餐？", "借問今仔日有啥特餐？"),
    RegressionCase("ordering", "請問有什麼推薦的嗎？", "借問有啥推薦的無？"),
    RegressionCase("ordering", "我要點這個。", "我欲點這个。"),
    RegressionCase("ordering", "他在點菜。", "伊佇咧叫菜。"),
    RegressionCase("ordering", "我想加點一份薯條。", "我想欲加點一份薯條。"),
    RegressionCase("ordering", "可以換成套餐嗎？", "會當換做套餐無？"),
    RegressionCase("ordering", "這份可以做外帶嗎？", "這份會當做外帶無？"),
    RegressionCase("ordering", "飲料可以去冰嗎？", "飲料會當去冰無？"),
    RegressionCase("ordering", "可以先上飲料嗎？", "會當先送飲料來無？"),
    RegressionCase("ordering", "可以幫我換成外帶嗎？", "會當替我換做外帶無？"),
    RegressionCase("ordering", "我想取消訂位。", "我想欲取消訂位。"),
    RegressionCase("ordering", "我需要取消訂位。", "我欲取消訂位。"),
    RegressionCase("ordering", "我想改訂位時間。", "我想欲改訂位時間。"),
    RegressionCase("ordering", "我需要改訂位時間。", "我欲改訂位時間。"),
    RegressionCase("ordering", "可以幫我取消訂位嗎？", "會當替我取消訂位無？"),
    RegressionCase("ordering", "請協助取消訂位。", "請鬥相共取消訂位。"),
    RegressionCase("ordering", "我想改成內用。", "我想欲改做內用。"),
    RegressionCase("ordering", "可以幫我查訂位紀錄嗎？", "會當替我查訂位紀錄無？"),
    RegressionCase("ordering", "我需要查訂位紀錄。", "我欲查訂位紀錄。"),
    RegressionCase("ordering", "我需要確認訂位時間。", "我欲確認訂位時間。"),
    RegressionCase("ordering", "我想確認訂位時間。", "我想欲確認訂位時間。"),
    RegressionCase("ordering", "可以幫我確認訂位時間嗎？", "會當替我確認訂位時間無？"),
    RegressionCase("ordering", "請幫我確認訂位時間。", "請替我確認訂位時間。"),
    RegressionCase("ordering", "麻煩你幫我確認訂位時間。", "麻煩你替我確認訂位時間。"),
    RegressionCase("ordering", "幫我確認訂位時間。", "替我確認訂位時間。"),
    RegressionCase("ordering", "請協助確認訂位時間。", "請鬥相共確認訂位時間。"),
    RegressionCase("ordering", "請協助查訂位時間。", "請鬥相共查訂位時間。"),
    RegressionCase("ordering", "可以幫我改訂位時間嗎？", "會當替我改訂位時間無？"),
    RegressionCase("ordering", "請協助改訂位時間。", "請鬥相共改訂位時間。"),
    RegressionCase("ordering", "我想改用外帶。", "我想欲改做外帶。"),
    RegressionCase("ordering", "可以幫我改用內用嗎？", "會當替我改做內用無？"),
    RegressionCase("ordering", "可以幫我查套餐內容嗎？", "會當替我查套餐內容無？"),
    RegressionCase("ordering", "可以幫我查訂單內容嗎？", "會當替我查訂單內容無？"),
    RegressionCase("ordering", "可以幫我查菜單內容嗎？", "會當替我查菜單內容無？"),
    RegressionCase("ordering", "可以幫我查飲料內容嗎？", "會當替我查飲料內容無？"),
    # spice / dietary — 口味偏好
    RegressionCase("spice_dietary", "不要太辣。", "莫太辣。"),
    RegressionCase("spice_dietary", "我不要加辣。", "我無愛加辣。"),
    RegressionCase("spice_dietary", "辣的還是不辣的？", "辣的猶是無辣的？"),
    RegressionCase("spice_dietary", "有沒有素食？", "敢有素食？"),
    RegressionCase("spice_dietary", "不要香菜。", "莫芫荽。"),
    RegressionCase("spice_dietary", "可以少鹽嗎？", "會當少鹽無？"),
    RegressionCaseModel(
        "spice_dietary",
        "我對花生過敏。",
        "我對塗豆過敏。",
        oracle_kind="ai_semantic_review",
        provenance="round531 AI semantic review; preserved general allergy status rather than narrowing it to ingestion",
        reviewed_by="Codex AI semantic review",
        reviewed_at="2026-07-28",
    ),
    RegressionCase("spice_dietary", "我不能吃牛肉。", "我袂當食牛肉。"),
    RegressionCase("spice_dietary", "可以不要放蔥嗎？", "會當免放蔥無？"),
    RegressionCase("spice_dietary", "這個可以不要辣嗎？", "這个會當免辣無？"),
    RegressionCase("spice_dietary", "可以幫我查素食選項嗎？", "會當替我查素食選項無？"),
    # seating — 座位
    RegressionCase("seating", "請問還有位子嗎？", "借問閣有位子無？"),
    RegressionCase("seating", "兩位大人一位小孩。", "兩位大人一位囡仔。"),
    RegressionCase("seating", "請問要等多久？", "借問愛等偌久？"),
    RegressionCase("seating", "我有訂位。", "我有訂位。"),
    RegressionCase("seating", "可以幫我安排兒童椅嗎？", "會當幫我安排囡仔椅無？"),
    RegressionCase("seating", "可以坐靠窗的位置嗎？", "會當坐靠窗的位無？"),
    RegressionCase("seating", "可以坐靠窗嗎？", "會當坐靠窗無？"),
    RegressionCase("seating", "需要等位嗎？", "需要等位無？"),
    RegressionCase("seating", "可以併桌嗎？", "會當併桌無？"),
    RegressionCase("seating", "有四個人的位子嗎？", "有四个人的位子無？"),
    RegressionCase("seating", "可以換到裡面的位置嗎？", "會當換到內底的位無？"),
    RegressionCase("seating", "可以幫我換座位嗎？", "會當替我換座位無？"),
    RegressionCase("seating", "我需要換座位。", "我欲換座位。"),
    RegressionCase("seating", "可以幫我查候位進度嗎？", "會當替我查候位進度無？"),
    RegressionCase("seating", "可以幫我查訂位人數嗎？", "會當替我查訂位人數無？"),
    RegressionCase("seating", "可以幫我查包廂座位嗎？", "會當替我查包廂座位無？"),
    RegressionCase("seating", "可以幫我查低消規定嗎？", "會當替我查低消規定無？"),
    # payment — 結帳
    RegressionCase("payment", "麻煩結帳。", "麻煩結數。"),
    RegressionCase("payment", "總共多少錢？", "總共偌濟錢？"),
    RegressionCase("payment", "這個多少錢？", "這个偌濟錢？"),
    RegressionCase(
        "payment",
        "可以刷卡嗎？",
        "會當刷卡無？",
        duplicate_group="cross_domain_payment_intent",
        duplicate_reason="餐飲、購物與計程車產品面共用付款意圖，但需各自保留端到端覆蓋。",
    ),
    RegressionCase("payment", "可以分開結帳嗎？", "會當分開結數無？"),
    RegressionCase("payment", "我要用現金付。", "我欲付現錢。"),
    RegressionCase(
        "payment",
        "可以開發票嗎？",
        "會當開發票無？",
        duplicate_group="cross_domain_payment_intent",
        duplicate_reason="餐飲、購物與計程車產品面共用付款意圖，但需各自保留端到端覆蓋。",
    ),
    RegressionCase("payment", "可以幫我開發票嗎？", "會當替我開發票無？"),
    RegressionCase("payment", "發票可以用載具嗎？", "發票會當用載具無？"),
    RegressionCase("payment", "我想改用信用卡付款。", "我想欲改用信用卡付款。"),
    RegressionCase("payment", "我想改用現金付款。", "我想欲改用現錢付款。"),
    RegressionCase("payment", "我想改用餐時間。", "我想欲改用餐時間。"),
    # service — 服務需求
    RegressionCase("service", "可以打包嗎？", "會當打包無？"),
    RegressionCase("service", "我要外帶。", "我欲外帶。"),
    RegressionCase("service", "可以幫我換盤子嗎？", "會當幫我換盤仔無？"),
    RegressionCase("service", "請問廁所在哪裡？", "借問便所佇佗位？"),
    RegressionCase("service", "可以加水嗎？", "會當加水無？"),
    RegressionCase("service", "可以換餐具嗎？", "會當換餐具無？"),
    RegressionCase("service", "我的餐還沒來。", "我的餐猶未來。"),
    RegressionCase("service", "可以加一張椅子嗎？", "會當加一張椅仔無？"),
    RegressionCase("service", "餐點可以快一點嗎？", "餐點會當較緊無？"),
    RegressionCase("service", "可以幫我加飯嗎？", "會當幫我添飯無？"),
    RegressionCase("service", "可以幫我拿餐具嗎？", "會當幫我提餐具來無？"),
    RegressionCase("service", "可以幫我加湯嗎？", "會當幫我添湯無？"),
    RegressionCase("service", "這道菜可以快一點嗎？", "這道菜會當較緊無？"),
    RegressionCase("service", "可以幫我拿吸管嗎？", "會當幫我提吸管來無？"),
    RegressionCase("service", "可以幫我加醬嗎？", "會當幫我添醬無？"),
    RegressionCase("service", "可以幫我拿衛生紙嗎？", "會當幫我提衛生紙來無？"),
    RegressionCase("service", "可以幫我換小碗嗎？", "會當幫我換細碗無？"),
    RegressionCase("service", "可以幫我查外送進度嗎？", "會當替我查外送進度無？"),
    RegressionCase("service", "可以幫我查餐點狀態嗎？", "會當替我查餐點狀態無？"),
    RegressionCase("service", "可以幫我查取餐時間嗎？", "會當替我查取餐時間無？"),
    RegressionCase("service", "可以幫我查桌號嗎？", "會當替我查桌號無？"),
    # round524 / AI semantic blind pilot (kept separate from human verification)
    RegressionCaseModel(
        "ai_semantic_pilot_food",
        "這碗湯太燙了，先放涼再喝。",
        "這碗湯太燙矣，先囥冷再啉。",
        oracle_kind="ai_semantic_review",
        provenance="100-case zero-overlap blind pilot; MOE dictionary example supports 囥予冷 for cooling food",
        reviewed_by="codex_ai_semantic_audit",
        reviewed_at="2026-07-28",
    ),
    RegressionCaseModel(
        "ai_semantic_pilot_food",
        "咖啡不要糖，牛奶少一點。",
        "咖啡毋愛糖，牛奶少一寡。",
        oracle_kind="ai_semantic_review",
        provenance="100-case zero-overlap blind pilot; ordering preference reviewed as 毋愛 rather than prohibitive 莫",
        reviewed_by="codex_ai_semantic_audit",
        reviewed_at="2026-07-28",
    ),
    RegressionCaseModel(
        "ai_semantic_pilot_food",
        "你要內用還是外帶？",
        "你欲內用抑是外帶？",
        oracle_kind="ai_semantic_review",
        provenance="independent 100-case output review; choice 還是 maps to 抑是 rather than continuative 猶是",
        reviewed_by="codex_and_independent_subagent",
        reviewed_at="2026-07-28",
    ),
    # round532 / authoritative low-trust machine cleanup
    RegressionCaseModel(
        "round532_authoritative_lexicon",
        "鹹稀飯",
        "鹹糜",
        oracle_kind="ai_semantic_review",
        provenance="round532 low-trust risk audit; AI semantic review cross-checked against the MOE Taiwanese Hokkien dictionary",
        reviewed_by="codex_ai_semantic_audit",
        reviewed_at="2026-07-29",
    ),
]


def main() -> int:
    return run_regression_cli(
        RESTAURANT_REGRESSION_CASES,
        description="餐廳點餐情境 regression runner",
    )


if __name__ == "__main__":
    raise SystemExit(main())
