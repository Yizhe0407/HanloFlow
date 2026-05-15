from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from converter import TaigiConverter


@dataclass(frozen=True)
class RegressionCase:
    category: str
    source: str
    expected: str


WORKPLACE_REGRESSION_CASES: list[RegressionCase] = [
    # meeting
    RegressionCase("meeting", "我今天要開會。", "我今仔日欲開會。"),
    RegressionCase("meeting", "下午要開會。", "下晝欲開會。"),
    RegressionCase("meeting", "會議改到明天。", "會議改做明仔載。"),
    RegressionCase("meeting", "明天早上再討論。", "明仔早起閣討論。"),
    RegressionCase("meeting", "會議取消了。", "會議取消矣。"),
    RegressionCase("meeting", "請協助取消會議。", "請鬥相共取消會議。"),
    RegressionCase("meeting", "會議延後到下午。", "會議延後到下晝。"),
    RegressionCase("meeting", "請把會議連結寄給我。", "請共會議連結寄予我。"),
    RegressionCase("meeting", "開會前提醒我。", "開會前共我提醒。"),
    RegressionCase("meeting", "主管說明天要開會。", "主管講明仔載欲開會。"),
    RegressionCase("meeting", "會議要準備資料嗎？", "會議愛準備資料無？"),
    RegressionCase("meeting", "請幫我確認會議時間。", "請替我確認會議時間。"),
    RegressionCase("meeting", "我需要確認會議時間。", "我欲確認會議時間。"),
    RegressionCase("meeting", "我想確認會議時間。", "我想欲確認會議時間。"),
    RegressionCase("meeting", "可以幫我確認會議時間嗎？", "會當替我確認會議時間無？"),
    RegressionCase("meeting", "麻煩你幫我確認會議時間。", "麻煩你替我確認會議時間。"),
    RegressionCase("meeting", "幫我確認會議時間。", "替我確認會議時間。"),
    RegressionCase("meeting", "請協助確認會議時間。", "請鬥相共確認會議時間。"),
    RegressionCase("meeting", "請協助查會議時間。", "請鬥相共查會議時間。"),
    RegressionCase("meeting", "可以改成線上會議嗎？", "會當改做線上會議無？"),
    RegressionCase("meeting", "我想改會議地點。", "我想欲改會議地點。"),
    RegressionCase("meeting", "可以幫我安排會議室嗎？", "會當替我安排會議室無？"),
    RegressionCase("meeting", "可以幫我預約會議室嗎？", "會當替我預約會議室無？"),
    RegressionCase("meeting", "我想改開會時間。", "我想欲改開會時間。"),
    RegressionCase("meeting", "可以幫我更新會議連結嗎？", "會當替我更新會議連結無？"),
    RegressionCase("meeting", "可以幫我寄會議紀錄嗎？", "會當替我寄會議紀錄無？"),
    RegressionCase("meeting", "我想改開會地點。", "我想欲改開會地點。"),
    RegressionCase("meeting", "我想改會議時間。", "我想欲改會議時間。"),
    RegressionCase("meeting", "請協助改會議時間。", "請鬥相共改會議時間。"),
    RegressionCase("meeting", "可以幫我查會議室設備嗎？", "會當替我查會議室設備無？"),
    RegressionCase("meeting", "可以幫我查會議進度嗎？", "會當替我查會議進度無？"),
    # office location / progressive
    RegressionCase("office_location", "主管在會議室。", "主管佇會議室。"),
    RegressionCase("office_location", "我們在會議室開會。", "咱佇會議室開會。"),
    RegressionCase("office_location", "資料放在桌上。", "資料囥佇桌頂。"),
    RegressionCase("office_location", "會議室在哪裡？", "會議室佇佗位？"),
    RegressionCase("office_location", "茶水間在走廊旁邊。", "茶水間佇走廊邊仔。"),
    RegressionCase("office_location", "影印機在櫃檯後面。", "影印機佇櫃檯後壁。"),
    RegressionCase("office_location", "我的座位在窗戶旁邊。", "我的座位佇窗仔門邊仔。"),
    RegressionCase("progressive", "同事在等你。", "同事佇咧等你。"),
    RegressionCase("progressive", "主管在看報告。", "主管佇咧看報告。"),
    RegressionCase("progressive", "我正在寫報告。", "我佇咧寫報告。"),
    RegressionCase("progressive", "我正在整理資料。", "我佇咧整理資料。"),
    RegressionCase("progressive", "他正在確認名單。", "伊佇咧確認名單。"),
    RegressionCase("progressive", "同事正在列印文件。", "同事佇咧列印文件。"),
    RegressionCase("progressive", "我們正在聯絡客戶。", "咱佇咧聯絡客戶。"),
    # workflow
    RegressionCase("workflow", "請你再確認一次。", "請你閣確認一擺。"),
    RegressionCase("workflow", "這份文件要簽名。", "這份文件愛簽名。"),
    RegressionCase("workflow", "報告還沒寫完。", "報告猶未寫完。"),
    RegressionCase("workflow", "請把檔案寄給我。", "請共檔案寄予我。"),
    RegressionCase("workflow", "請幫我上傳檔案。", "請替我上傳檔案。"),
    RegressionCase("workflow", "我等主管回覆。", "我等主管回覆。"),
    RegressionCase("workflow", "資料要改一下。", "資料愛改一下。"),
    RegressionCase("workflow", "請把新版寄給客戶。", "請共新版寄予客戶。"),
    RegressionCase("workflow", "我可以晚一點交報告嗎？", "我會當較晏一點交報告無？"),
    RegressionCase("workflow", "報告要在下班前交。", "報告愛佇下班前交。"),
    RegressionCase("workflow", "請幫我轉給主管。", "請替我轉予主管。"),
    RegressionCase("workflow", "這份資料要更新。", "這份資料愛更新。"),
    RegressionCase("workflow", "資料我晚點補。", "資料我較晏補。"),
    RegressionCase("workflow", "請幫我看一下這份資料。", "請替我看覓這份資料。"),
    RegressionCase("workflow", "請幫我追一下進度。", "請替我追一下進度。"),
    RegressionCase("workflow", "可以幫我列印文件嗎？", "會當替我列印文件無？"),
    RegressionCase("workflow", "我想改報告期限。", "我想欲改報告期限。"),
    RegressionCase("workflow", "可以幫我查簽核進度嗎？", "會當替我查簽核進度無？"),
    RegressionCase("workflow", "可以幫我查報表進度嗎？", "會當替我查報表進度無？"),
    RegressionCase("workflow", "可以幫我查合約進度嗎？", "會當替我查合約進度無？"),
    RegressionCase("workflow", "可以幫我查報銷進度嗎？", "會當替我查報銷進度無？"),
    RegressionCase("workflow", "可以幫我查請款狀態嗎？", "會當替我查請款狀態無？"),
    RegressionCase("workflow", "可以幫我查出差申請嗎？", "會當替我查出張申請無？"),
    RegressionCase("workflow", "請協助申請出差。", "請鬥相共申請出張。"),
    RegressionCase("workflow", "可以幫我查專案進度嗎？", "會當替我查專案進度無？"),
    RegressionCase("workflow", "可以幫我查任務進度嗎？", "會當替我查任務進度無？"),
    RegressionCase("workflow", "可以幫我查薪資明細嗎？", "會當替我查月給明細無？"),
    RegressionCase("workflow", "可以幫課員換班嗎？", "會當替課員換班無？"),
    RegressionCase("workflow", "可不可以幫課員換班？", "敢會當替課員換班？"),
    RegressionCase("workflow", "請協助課員換班。", "請鬥相共課員換班。"),
    RegressionCase("workflow", "可以幫科員換班嗎？", "會當替科員換班無？"),
    RegressionCase("workflow", "可不可以幫科員換班？", "敢會當替科員換班？"),
    RegressionCase("workflow", "請協助科員換班。", "請鬥相共科員換班。"),
    RegressionCase("workflow", "可以幫科長換班嗎？", "會當替科長換班無？"),
    RegressionCase("workflow", "可不可以幫科長換班？", "敢會當替科長換班？"),
    RegressionCase("workflow", "請協助科長換班。", "請鬥相共科長換班。"),
    RegressionCase("workflow", "可以幫總務換班嗎？", "會當替總務換班無？"),
    RegressionCase("workflow", "可不可以幫總務換班？", "敢會當替總務換班？"),
    RegressionCase("workflow", "請協助總務換班。", "請鬥相共總務換班。"),
    RegressionCase("workflow", "可以幫技師換班嗎？", "會當替技師換班無？"),
    RegressionCase("workflow", "可不可以幫技師換班？", "敢會當替技師換班？"),
    RegressionCase("workflow", "請協助技師換班。", "請鬥相共技師換班。"),
    RegressionCase("workflow", "可以幫師傅換班嗎？", "會當替師傅換班無？"),
    RegressionCase("workflow", "可不可以幫師傅換班？", "敢會當替師傅換班？"),
    RegressionCase("workflow", "請協助師傅換班。", "請鬥相共師傅換班。"),
    RegressionCase("workflow", "可以幫書記換班嗎？", "會當替書記換班無？"),
    RegressionCase("workflow", "可不可以幫書記換班？", "敢會當替書記換班？"),
    RegressionCase("workflow", "請協助書記換班。", "請鬥相共書記換班。"),
    RegressionCase("workflow", "可以幫主任秘書換班嗎？", "會當替主任秘書換班無？"),
    RegressionCase("workflow", "可不可以幫主任秘書換班？", "敢會當替主任秘書換班？"),
    RegressionCase("workflow", "請協助主任秘書換班。", "請鬥相共主任秘書換班。"),
    RegressionCase("workflow", "可以幫秘書長換班嗎？", "會當替秘書長換班無？"),
    RegressionCase("workflow", "可不可以幫秘書長換班？", "敢會當替秘書長換班？"),
    RegressionCase("workflow", "請協助秘書長換班。", "請鬥相共秘書長換班。"),
    RegressionCase("workflow", "可以幫工程師換班嗎？", "會當替工程師換班無？"),
    RegressionCase("workflow", "可不可以幫工程師換班？", "敢會當替工程師換班？"),
    RegressionCase("workflow", "請協助工程師換班。", "請鬥相共工程師換班。"),
    RegressionCase("workflow", "可以幫企劃換班嗎？", "會當替企劃換班無？"),
    RegressionCase("workflow", "可不可以幫企劃換班？", "敢會當替企劃換班？"),
    RegressionCase("workflow", "請協助企劃換班。", "請鬥相共企劃換班。"),
    RegressionCase("workflow", "可以幫顧問換班嗎？", "會當替顧問換班無？"),
    RegressionCase("workflow", "可不可以幫顧問換班？", "敢會當替顧問換班？"),
    RegressionCase("workflow", "請協助顧問換班。", "請鬥相共顧問換班。"),
    RegressionCase("workflow", "可以幫行政人員換班嗎？", "會當替行政人員換班無？"),
    RegressionCase("workflow", "可不可以幫行政人員換班？", "敢會當替行政人員換班？"),
    RegressionCase("workflow", "請協助行政人員換班。", "請鬥相共行政人員換班。"),
    RegressionCase("workflow", "可以幫人事人員換班嗎？", "會當替人事人員換班無？"),
    RegressionCase("workflow", "可不可以幫人事人員換班？", "敢會當替人事人員換班？"),
    RegressionCase("workflow", "請協助人事人員換班。", "請鬥相共人事人員換班。"),
    RegressionCase("workflow", "可以幫會計人員換班嗎？", "會當替會計人員換班無？"),
    RegressionCase("workflow", "可不可以幫會計人員換班？", "敢會當替會計人員換班？"),
    RegressionCase("workflow", "請協助會計人員換班。", "請鬥相共會計人員換班。"),
    RegressionCase("workflow", "可以幫出納人員換班嗎？", "會當替出納人員換班無？"),
    RegressionCase("workflow", "可不可以幫出納人員換班？", "敢會當替出納人員換班？"),
    RegressionCase("workflow", "請協助出納人員換班。", "請鬥相共出納人員換班。"),
    RegressionCase("workflow", "可以幫採購人員換班嗎？", "會當替採購人員換班無？"),
    RegressionCase("workflow", "可不可以幫採購人員換班？", "敢會當替採購人員換班？"),
    RegressionCase("workflow", "請協助採購人員換班。", "請鬥相共採購人員換班。"),
    RegressionCase("workflow", "可以幫工務人員換班嗎？", "會當替工務人員換班無？"),
    RegressionCase("workflow", "可不可以幫工務人員換班？", "敢會當替工務人員換班？"),
    RegressionCase("workflow", "請協助工務人員換班。", "請鬥相共工務人員換班。"),
    RegressionCase("workflow", "可以幫品保人員換班嗎？", "會當替品保人員換班無？"),
    RegressionCase("workflow", "可不可以幫品保人員換班？", "敢會當替品保人員換班？"),
    RegressionCase("workflow", "請協助品保人員換班。", "請鬥相共品保人員換班。"),
    RegressionCase("workflow", "可以幫品管人員換班嗎？", "會當替品管人員換班無？"),
    RegressionCase("workflow", "可不可以幫品管人員換班？", "敢會當替品管人員換班？"),
    RegressionCase("workflow", "請協助品管人員換班。", "請鬥相共品管人員換班。"),
    RegressionCase("workflow", "可以幫業務人員換班嗎？", "會當替業務人員換班無？"),
    RegressionCase("workflow", "可不可以幫業務人員換班？", "敢會當替業務人員換班？"),
    RegressionCase("workflow", "請協助業務人員換班。", "請鬥相共業務人員換班。"),
    RegressionCase("workflow", "可以幫倉管人員換班嗎？", "會當替倉管人員換班無？"),
    RegressionCase("workflow", "可不可以幫倉管人員換班？", "敢會當替倉管人員換班？"),
    RegressionCase("workflow", "請協助倉管人員換班。", "請鬥相共倉管人員換班。"),
    RegressionCase("workflow", "可以幫保全人員換班嗎？", "會當替保全人員換班無？"),
    RegressionCase("workflow", "可不可以幫保全人員換班？", "敢會當替保全人員換班？"),
    RegressionCase("workflow", "請協助保全人員換班。", "請鬥相共保全人員換班。"),
    RegressionCase("workflow", "可以幫研發人員換班嗎？", "會當替研發人員換班無？"),
    RegressionCase("workflow", "可不可以幫研發人員換班？", "敢會當替研發人員換班？"),
    RegressionCase("workflow", "請協助研發人員換班。", "請鬥相共研發人員換班。"),
    RegressionCase("workflow", "可以幫包裝人員換班嗎？", "會當替包裝人員換班無？"),
    RegressionCase("workflow", "可不可以幫包裝人員換班？", "敢會當替包裝人員換班？"),
    RegressionCase("workflow", "請協助包裝人員換班。", "請鬥相共包裝人員換班。"),
    RegressionCase("workflow", "可以幫物流人員換班嗎？", "會當替物流人員換班無？"),
    RegressionCase("workflow", "可不可以幫物流人員換班？", "敢會當替物流人員換班？"),
    RegressionCase("workflow", "請協助物流人員換班。", "請鬥相共物流人員換班。"),
    RegressionCase("workflow", "可以幫維修人員換班嗎？", "會當替維修人員換班無？"),
    RegressionCase("workflow", "可不可以幫維修人員換班？", "敢會當替維修人員換班？"),
    RegressionCase("workflow", "請協助維修人員換班。", "請鬥相共維修人員換班。"),
    RegressionCase("workflow", "可以幫作業員換班嗎？", "會當替作業員換班無？"),
    RegressionCase("workflow", "可不可以幫作業員換班？", "敢會當替作業員換班？"),
    RegressionCase("workflow", "請協助作業員換班。", "請鬥相共作業員換班。"),
    RegressionCase("workflow", "可以幫操作員換班嗎？", "會當替操作員換班無？"),
    RegressionCase("workflow", "可不可以幫操作員換班？", "敢會當替操作員換班？"),
    RegressionCase("workflow", "請協助操作員換班。", "請鬥相共操作員換班。"),
    RegressionCase("workflow", "可以幫客服專員換班嗎？", "會當替客服專員換班無？"),
    RegressionCase("workflow", "可不可以幫客服專員換班？", "敢會當替客服專員換班？"),
    RegressionCase("workflow", "請協助客服專員換班。", "請鬥相共客服專員換班。"),
    RegressionCase("workflow", "可以幫客服主管換班嗎？", "會當替客服主管換班無？"),
    RegressionCase("workflow", "可不可以幫客服主管換班？", "敢會當替客服主管換班？"),
    RegressionCase("workflow", "請協助客服主管換班。", "請鬥相共客服主管換班。"),
    RegressionCase("workflow", "可以幫專案經理換班嗎？", "會當替專案經理換班無？"),
    RegressionCase("workflow", "可不可以幫專案經理換班？", "敢會當替專案經理換班？"),
    RegressionCase("workflow", "請協助專案經理換班。", "請鬥相共專案經理換班。"),
    RegressionCase("workflow", "可以幫資深工程師換班嗎？", "會當替資深工程師換班無？"),
    RegressionCase("workflow", "可不可以幫資深工程師換班？", "敢會當替資深工程師換班？"),
    RegressionCase("workflow", "請協助資深工程師換班。", "請鬥相共資深工程師換班。"),
    RegressionCase("workflow", "可以幫設計師換班嗎？", "會當替設計師換班無？"),
    RegressionCase("workflow", "可不可以幫設計師換班？", "敢會當替設計師換班？"),
    RegressionCase("workflow", "請協助設計師換班。", "請鬥相共設計師換班。"),
    RegressionCase("workflow", "可以幫前台人員換班嗎？", "會當替前台人員換班無？"),
    RegressionCase("workflow", "可不可以幫前台人員換班？", "敢會當替前台人員換班？"),
    RegressionCase("workflow", "請協助前台人員換班。", "請鬥相共前台人員換班。"),
    RegressionCase("workflow", "可以幫接待人員換班嗎？", "會當替接待人員換班無？"),
    RegressionCase("workflow", "可不可以幫接待人員換班？", "敢會當替接待人員換班？"),
    RegressionCase("workflow", "請協助接待人員換班。", "請鬥相共接待人員換班。"),
    RegressionCase("workflow", "可以幫門市人員換班嗎？", "會當替門市人員換班無？"),
    RegressionCase("workflow", "可不可以幫門市人員換班？", "敢會當替門市人員換班？"),
    RegressionCase("workflow", "請協助門市人員換班。", "請鬥相共門市人員換班。"),
    RegressionCase("workflow", "可以幫餐飲人員換班嗎？", "會當替餐飲人員換班無？"),
    RegressionCase("workflow", "可不可以幫餐飲人員換班？", "敢會當替餐飲人員換班？"),
    RegressionCase("workflow", "請協助餐飲人員換班。", "請鬥相共餐飲人員換班。"),
    RegressionCase("workflow", "可以幫洗碗工換班嗎？", "會當替洗碗工換班無？"),
    RegressionCase("workflow", "可不可以幫洗碗工換班？", "敢會當替洗碗工換班？"),
    RegressionCase("workflow", "請協助洗碗工換班。", "請鬥相共洗碗工換班。"),
    RegressionCase("workflow", "可以幫售票員換班嗎？", "會當替售票員換班無？"),
    RegressionCase("workflow", "可不可以幫售票員換班？", "敢會當替售票員換班？"),
    RegressionCase("workflow", "請協助售票員換班。", "請鬥相共售票員換班。"),
    RegressionCase("workflow", "可以幫櫃員換班嗎？", "會當替櫃員換班無？"),
    RegressionCase("workflow", "可不可以幫櫃員換班？", "敢會當替櫃員換班？"),
    RegressionCase("workflow", "請協助櫃員換班。", "請鬥相共櫃員換班。"),
    RegressionCase("workflow", "可以幫門衛換班嗎？", "會當替門衛換班無？"),
    RegressionCase("workflow", "可不可以幫門衛換班？", "敢會當替門衛換班？"),
    RegressionCase("workflow", "請協助門衛換班。", "請鬥相共門衛換班。"),
    RegressionCase("workflow", "可以幫警衛換班嗎？", "會當替警衛換班無？"),
    RegressionCase("workflow", "可不可以幫警衛換班？", "敢會當替警衛換班？"),
    RegressionCase("workflow", "請協助警衛換班。", "請鬥相共警衛換班。"),
    RegressionCase("workflow", "可以幫護工換班嗎？", "會當替護工換班無？"),
    RegressionCase("workflow", "可不可以幫護工換班？", "敢會當替護工換班？"),
    RegressionCase("workflow", "請協助護工換班。", "請鬥相共護工換班。"),
    RegressionCase("workflow", "可以幫編輯換班嗎？", "會當替編輯換班無？"),
    RegressionCase("workflow", "可不可以幫編輯換班？", "敢會當替編輯換班？"),
    RegressionCase("workflow", "請協助編輯換班。", "請鬥相共編輯換班。"),
    RegressionCase("workflow", "可以幫主播換班嗎？", "會當替主播換班無？"),
    RegressionCase("workflow", "可不可以幫主播換班？", "敢會當替主播換班？"),
    RegressionCase("workflow", "請協助主播換班。", "請鬥相共主播換班。"),
    RegressionCase("workflow", "可以幫前台人員換班嗎？", "會當替前台人員換班無？"),
    RegressionCase("workflow", "可不可以幫前台人員換班？", "敢會當替前台人員換班？"),
    RegressionCase("workflow", "請協助前台人員換班。", "請鬥相共前台人員換班。"),
    RegressionCase("workflow", "可以幫接待人員換班嗎？", "會當替接待人員換班無？"),
    RegressionCase("workflow", "可不可以幫接待人員換班？", "敢會當替接待人員換班？"),
    RegressionCase("workflow", "請協助接待人員換班。", "請鬥相共接待人員換班。"),
    RegressionCase("workflow", "可以幫門市店員換班嗎？", "會當替門市店員換班無？"),
    RegressionCase("workflow", "可不可以幫門市店員換班？", "敢會當替門市店員換班？"),
    RegressionCase("workflow", "請協助門市店員換班。", "請鬥相共門市店員換班。"),
    RegressionCase("workflow", "可以幫客服代表換班嗎？", "會當替客服代表換班無？"),
    RegressionCase("workflow", "可不可以幫客服代表換班？", "敢會當替客服代表換班？"),
    RegressionCase("workflow", "請協助客服代表換班。", "請鬥相共客服代表換班。"),
    RegressionCase("workflow", "可以幫門衛換班嗎？", "會當替門衛換班無？"),
    RegressionCase("workflow", "可不可以幫門衛換班？", "敢會當替門衛換班？"),
    RegressionCase("workflow", "請協助門衛換班。", "請鬥相共門衛換班。"),
    RegressionCase("workflow", "可以幫警衛換班嗎？", "會當替警衛換班無？"),
    RegressionCase("workflow", "可不可以幫警衛換班？", "敢會當替警衛換班？"),
    RegressionCase("workflow", "請協助警衛換班。", "請鬥相共警衛換班。"),
    RegressionCase("workflow", "可以幫保姆換班嗎？", "會當替保姆換班無？"),
    RegressionCase("workflow", "可不可以幫保姆換班？", "敢會當替保姆換班？"),
    RegressionCase("workflow", "請協助保姆換班。", "請鬥相共保姆換班。"),
    RegressionCase("workflow", "可以幫護工換班嗎？", "會當替護工換班無？"),
    RegressionCase("workflow", "可不可以幫護工換班？", "敢會當替護工換班？"),
    RegressionCase("workflow", "請協助護工換班。", "請鬥相共護工換班。"),
    RegressionCase("workflow", "可以幫記者換班嗎？", "會當替記者換班無？"),
    RegressionCase("workflow", "可不可以幫記者換班？", "敢會當替記者換班？"),
    RegressionCase("workflow", "請協助記者換班。", "請鬥相共記者換班。"),
    RegressionCase("workflow", "可以幫編輯換班嗎？", "會當替編輯換班無？"),
    RegressionCase("workflow", "可不可以幫編輯換班？", "敢會當替編輯換班？"),
    RegressionCase("workflow", "請協助編輯換班。", "請鬥相共編輯換班。"),
    RegressionCase("workflow", "可以幫主播換班嗎？", "會當替主播換班無？"),
    RegressionCase("workflow", "可不可以幫主播換班？", "敢會當替主播換班？"),
    RegressionCase("workflow", "請協助主播換班。", "請鬥相共主播換班。"),
    RegressionCase("workflow", "可以幫總監換班嗎？", "會當替總監換班無？"),
    RegressionCase("workflow", "可不可以幫總監換班？", "敢會當替總監換班？"),
    RegressionCase("workflow", "請協助總監換班。", "請鬥相共總監換班。"),
    RegressionCase("workflow", "可以幫專員換班嗎？", "會當替專員換班無？"),
    RegressionCase("workflow", "可不可以幫專員換班？", "敢會當替專員換班？"),
    RegressionCase("workflow", "請協助專員換班。", "請鬥相共專員換班。"),
    RegressionCase("workflow", "可以幫副主任換班嗎？", "會當替副主任換班無？"),
    RegressionCase("workflow", "可不可以幫副主任換班？", "敢會當替副主任換班？"),
    RegressionCase("workflow", "請協助副主任換班。", "請鬥相共副主任換班。"),
    RegressionCase("workflow", "可以幫主任換班嗎？", "會當替主任換班無？"),
    RegressionCase("workflow", "可不可以幫主任換班？", "敢會當替主任換班？"),
    RegressionCase("workflow", "請協助主任換班。", "請鬥相共主任換班。"),
    RegressionCase("workflow", "可以幫組長換班嗎？", "會當替組長換班無？"),
    RegressionCase("workflow", "可不可以幫組長換班？", "敢會當替組長換班？"),
    RegressionCase("workflow", "請協助組長換班。", "請鬥相共組長換班。"),
    RegressionCase("workflow", "可以幫助理換班嗎？", "會當替助理換班無？"),
    RegressionCase("workflow", "可不可以幫助理換班？", "敢會當替助理換班？"),
    RegressionCase("workflow", "請協助助理換班。", "請鬥相共助理換班。"),
    RegressionCase("workflow", "可以幫課長換班嗎？", "會當替課長換班無？"),
    RegressionCase("workflow", "可不可以幫課長換班？", "敢會當替課長換班？"),
    RegressionCase("workflow", "請協助課長換班。", "請鬥相共課長換班。"),
    RegressionCase("workflow", "可以幫清潔人員換班嗎？", "會當替清掃人員換班無？"),
    RegressionCase("workflow", "可不可以幫清潔人員換班？", "敢會當替清掃人員換班？"),
    RegressionCase("workflow", "請協助清潔人員換班。", "請鬥相共清掃人員換班。"),
    RegressionCase("workflow", "可以幫護理師換班嗎？", "會當替護理師換班無？"),
    RegressionCase("workflow", "可不可以幫護理師換班？", "敢會當替護理師換班？"),
    RegressionCase("workflow", "請協助護理師換班。", "請鬥相共護理師換班。"),
    RegressionCase("workflow", "可以幫護士換班嗎？", "會當替護理師換班無？"),
    RegressionCase("workflow", "可不可以幫護士換班？", "敢會當替護理師換班？"),
    RegressionCase("workflow", "請協助護士換班。", "請鬥相共護理師換班。"),
    RegressionCase("workflow", "可以幫客服人員換班嗎？", "會當替客服人員換班無？"),
    RegressionCase("workflow", "可不可以幫客服人員換班？", "敢會當替客服人員換班？"),
    RegressionCase("workflow", "請協助客服人員換班。", "請鬥相共客服人員換班。"),
    RegressionCase("workflow", "可以幫服務人員換班嗎？", "會當替服務人員換班無？"),
    RegressionCase("workflow", "可不可以幫服務人員換班？", "敢會當替服務人員換班？"),
    RegressionCase("workflow", "請協助服務人員換班。", "請鬥相共服務人員換班。"),
    RegressionCase("workflow", "可以幫櫃檯人員換班嗎？", "會當替櫃檯人員換班無？"),
    RegressionCase("workflow", "可不可以幫櫃檯人員換班？", "敢會當替櫃檯人員換班？"),
    RegressionCase("workflow", "請協助櫃檯人員換班。", "請鬥相共櫃檯人員換班。"),
    RegressionCase("workflow", "可以幫櫃台人員換班嗎？", "會當替櫃檯人員換班無？"),
    RegressionCase("workflow", "可不可以幫櫃台人員換班？", "敢會當替櫃檯人員換班？"),
    RegressionCase("workflow", "請協助櫃台人員換班。", "請鬥相共櫃檯人員換班。"),
    RegressionCase("workflow", "可以幫藥師換班嗎？", "會當替藥師換班無？"),
    RegressionCase("workflow", "可不可以幫藥師換班？", "敢會當替藥師換班？"),
    RegressionCase("workflow", "請協助藥師換班。", "請鬥相共藥師換班。"),
    RegressionCase("workflow", "可以幫藥劑師換班嗎？", "會當替藥劑師換班無？"),
    RegressionCase("workflow", "可不可以幫藥劑師換班？", "敢會當替藥劑師換班？"),
    RegressionCase("workflow", "請協助藥劑師換班。", "請鬥相共藥劑師換班。"),
    RegressionCase("workflow", "可以幫清掃人員換班嗎？", "會當替清掃人員換班無？"),
    RegressionCase("workflow", "可不可以幫清掃人員換班？", "敢會當替清掃人員換班？"),
    RegressionCase("workflow", "請協助清掃人員換班。", "請鬥相共清掃人員換班。"),
    RegressionCase("workflow", "可以幫同事換班嗎？", "會當替同事換班無？"),
    RegressionCase("workflow", "可不可以幫同事換班？", "敢會當替同事換班？"),
    RegressionCase("workflow", "請協助同事換班。", "請鬥相共同事換班。"),
    RegressionCase("workflow", "可以幫教練換班嗎？", "會當替教練換班無？"),
    RegressionCase("workflow", "可不可以幫教練換班？", "敢會當替教練換班？"),
    RegressionCase("workflow", "請協助教練換班。", "請鬥相共教練換班。"),
    RegressionCase("workflow", "可以幫家教換班嗎？", "會當替家教換班無？"),
    RegressionCase("workflow", "可不可以幫家教換班？", "敢會當替家教換班？"),
    RegressionCase("workflow", "請協助家教換班。", "請鬥相共家教換班。"),
    RegressionCase("workflow", "可以幫主管換班嗎？", "會當替主管換班無？"),
    RegressionCase("workflow", "可不可以幫主管換班？", "敢會當替主管換班？"),
    RegressionCase("workflow", "請協助主管換班。", "請鬥相共主管換班。"),
    RegressionCase("workflow", "可以幫經理換班嗎？", "會當替經理換班無？"),
    RegressionCase("workflow", "可不可以幫經理換班？", "敢會當替經理換班？"),
    RegressionCase("workflow", "請協助經理換班。", "請鬥相共經理換班。"),
    RegressionCase("workflow", "可以幫副理換班嗎？", "會當替副理換班無？"),
    RegressionCase("workflow", "可不可以幫副理換班？", "敢會當替副理換班？"),
    RegressionCase("workflow", "請協助副理換班。", "請鬥相共副理換班。"),
    RegressionCase("workflow", "可以幫店長換班嗎？", "會當替店長換班無？"),
    RegressionCase("workflow", "可不可以幫店長換班？", "敢會當替店長換班？"),
    RegressionCase("workflow", "請協助店長換班。", "請鬥相共店長換班。"),
    RegressionCase("workflow", "可以幫講師換班嗎？", "會當替講師換班無？"),
    RegressionCase("workflow", "可不可以幫講師換班？", "敢會當替講師換班？"),
    RegressionCase("workflow", "請協助講師換班。", "請鬥相共講師換班。"),
    RegressionCase("workflow", "可以幫研究員換班嗎？", "會當替研究員換班無？"),
    RegressionCase("workflow", "可不可以幫研究員換班？", "敢會當替研究員換班？"),
    RegressionCase("workflow", "請協助研究員換班。", "請鬥相共研究員換班。"),
    RegressionCase("workflow", "可以幫實習生換班嗎？", "會當替實習生換班無？"),
    RegressionCase("workflow", "可不可以幫實習生換班？", "敢會當替實習生換班？"),
    RegressionCase("workflow", "請協助實習生換班。", "請鬥相共實習生換班。"),
    RegressionCase("workflow", "可以幫教授換班嗎？", "會當替教授換班無？"),
    RegressionCase("workflow", "可不可以幫教授換班？", "敢會當替教授換班？"),
    RegressionCase("workflow", "請協助教授換班。", "請鬥相共教授換班。"),
    # leave / availability
    RegressionCase("leave_availability", "我今天請假。", "我今仔日請假。"),
    RegressionCase("leave_availability", "我想請半天假。", "我想欲請半日假。"),
    RegressionCase("leave_availability", "你有空嗎？", "你有閒無？"),
    RegressionCase("leave_availability", "我現在不方便。", "我這馬無方便。"),
    RegressionCase("leave_availability", "我今天會晚點到。", "我今仔日會較晏到。"),
    RegressionCase("leave_availability", "我想請病假。", "我想欲告病。"),
    RegressionCase("leave_availability", "我臨時有事。", "我臨時有代誌。"),
    RegressionCase("leave_availability", "我不在座位上。", "我無佇座位頂。"),
    RegressionCase("leave_availability", "我晚點進公司。", "我較晏進公司。"),
    RegressionCase("leave_availability", "可以幫我代班嗎？", "會當替我代班無？"),
    RegressionCase("leave_availability", "可以幫我改時間嗎？", "會當替我改時間無？"),
    RegressionCase("leave_availability", "可以幫我請公假嗎？", "會當替我請公假無？"),
    RegressionCase("leave_availability", "可以幫我登記請假嗎？", "會當替我登記請假無？"),
    RegressionCase("leave_availability", "請協助取消請假。", "請鬥相共取消請假。"),
    RegressionCase("leave_availability", "我想改班表。", "我想欲改班表。"),
    RegressionCase("leave_availability", "我想改上班時間。", "我想欲改上班時間。"),
    RegressionCase("leave_availability", "可以幫我查請假紀錄嗎？", "會當替我查請假紀錄無？"),
    RegressionCase("leave_availability", "可以幫我查加班申請嗎？", "會當替我查加班申請無？"),
    RegressionCase("leave_availability", "請協助申請加班。", "請鬥相共申請加班。"),
]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="辦公/會議情境 regression runner")
    parser.add_argument("--rounds", type=int, default=1, help="重複跑幾輪 exact-match 回歸")
    parser.add_argument("--category", action="append", default=[], help="只跑指定 category，可重複傳入")
    parser.add_argument("--list-categories", action="store_true", help="列出所有 category")
    parser.add_argument("--show-pass", action="store_true", help="顯示每筆通過案例")
    parser.add_argument("--fail-fast", action="store_true", help="遇到第一個 mismatch 就停止")
    return parser


def _selected_cases(categories: list[str]) -> list[RegressionCase]:
    if not categories:
        return WORKPLACE_REGRESSION_CASES
    wanted = set(categories)
    return [case for case in WORKPLACE_REGRESSION_CASES if case.category in wanted]


def main() -> int:
    args = _build_parser().parse_args()
    categories = sorted({case.category for case in WORKPLACE_REGRESSION_CASES})
    if args.list_categories:
        for category in categories:
            print(category)
        return 0

    cases = _selected_cases(args.category)
    if not cases:
        print("no cases selected")
        return 1

    converter = TaigiConverter()
    latencies_ms: list[float] = []
    category_counts = Counter(case.category for case in cases)

    print({"rounds": args.rounds, "case_count": len(cases), "categories": dict(sorted(category_counts.items()))})

    for round_idx in range(1, args.rounds + 1):
        failures: list[tuple[int, RegressionCase, str]] = []
        for index, case in enumerate(cases, 1):
            started = time.perf_counter()
            output = converter.convert(case.source)
            latencies_ms.append((time.perf_counter() - started) * 1000)
            if output != case.expected:
                failures.append((index, case, output))
                if args.fail_fast:
                    break
            elif args.show_pass:
                print(f"PASS round={round_idx} idx={index} category={case.category} src={case.source}")

        print({"round": round_idx, "failed": len(failures)})
        if failures:
            for index, case, output in failures[:10]:
                print(f"FAIL idx={index} category={case.category}")
                print(f"  src: {case.source}")
                print(f"  exp: {case.expected}")
                print(f"  out: {output}")
            return 1

    latencies_ms.sort()
    p95_index = max(int(len(latencies_ms) * 0.95) - 1, 0)
    print(
        {
            "status": "ok",
            "rounds": args.rounds,
            "case_count": len(cases),
            "total_checks": len(latencies_ms),
            "mean_ms": round(mean(latencies_ms), 4),
            "p95_ms": round(latencies_ms[p95_index], 4),
            "max_ms": round(max(latencies_ms), 4),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
