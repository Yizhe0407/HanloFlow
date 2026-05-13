from __future__ import annotations

import argparse
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from converter import TaigiConverter


@dataclass(frozen=True)
class RegressionCase:
    category: str
    source: str
    expected: str


BUS_REGRESSION_CASES: list[RegressionCase] = [
    # stop control / stop status
    RegressionCase("stop_control", "這班公車恢復停靠本站了，你不用再走去對面。", "這班公車閣停本站矣，你免閣行去對面。"),
    RegressionCase("stop_control", "這班公車只停外圍站，不會開進廟口前面。", "這班公車干焦停外圍站，袂開進廟埕頭前。"),
    RegressionCase("stop_control", "如果你要去醫院院區裡面，這班車沒有進去。", "若是你欲去病院內底，這班車無進去。"),
    RegressionCase("stop_control", "這個站牌已經恢復正常停靠了。", "這个站牌已經閣正常停矣。"),
    RegressionCase("stop_control", "這個站牌今天恢復停靠，但班次還沒有完全正常。", "這个站牌今仔日閣停，但班次猶未攏正常。"),
    RegressionCase("stop_control", "這個站牌暫時改成下車專用，上車請到對面。", "這个站牌暫時改成干焦落車，上車請到對面。"),
    RegressionCase("stop_control", "這一站今天暫停使用，請改到前面臨時站牌。", "這一站今仔日停用，請改去頭前臨時站牌。"),
    RegressionCase("stop_control", "這班公車今天不停靠學校門口。", "這班公車今仔日無停學校門跤口。"),
    RegressionCase("stop_control", "這班車今天不停靠北港朝天宮。", "這班車今仔日無停北港朝天宮。"),
    RegressionCase("stop_control", "這裡不是下車站，你要到前面那一站下。", "遮毋是落車站，你欲到頭前彼站才落。"),
    RegressionCase("stop_control", "這裡只能下車，不能上車。", "遮干焦會當落車，袂予人上車。"),
    RegressionCase("stop_control", "司機說這一站只下客，不給上車。", "司機講這一站干焦落客，袂予上車。"),
    RegressionCase("stop_control", "這站只讓下車，不讓上車。", "這站干焦落車，無予上車。"),
    RegressionCase("stop_control", "這班公車現在只下客，不載人上車。", "這班公車這馬干焦落客，無予人上車。"),
    RegressionCase("stop_control", "這班公車現在只停外圍站，你要自己走進去老街。", "這班公車這馬干焦停外圍站，你欲家己行入去老街。"),
    RegressionCase("stop_control", "今天因為活動，公車只停外圍，不會開進老街裡面。", "今仔日因為活動，公車干焦停外圍，袂開進老街內底。"),
    RegressionCase("stop_control", "這班車回程不會經過這一站。", "這班車回程袂經過這一站。"),
    RegressionCase("stop_control", "你要去對面搭回程車。", "你愛去對面搭回程車。"),
    RegressionCase("stop_control", "高鐵接駁車在對面站牌搭。", "高鐵接駁車佇對面站牌搭。"),
    RegressionCase("stop_control", "這班車今天只開前門。", "這班車今仔日干焦開前門。"),
    RegressionCase("stop_control", "後門現在不開放，請從前門上車。", "後門這馬無開放，請對頭前門上車。"),
    # delay / eta / detour
    RegressionCase("delay_eta", "如果你趕時間，我不建議你等這班車。", "若是你趕時間，我無建議你等這班車。"),
    RegressionCase("delay_eta", "如果你趕時間，建議你不要等這班車。", "若是你趕時間，建議你莫等這班車。"),
    RegressionCase("delay_eta", "因為前面塞車，現在沒有辦法給你很準的到站時間。", "因為頭前窒車，這馬無法度共你報真準的到站時間。"),
    RegressionCase("delay_eta", "前面有交通事故，公車會慢十五分鐘。", "頭前有交通事故，公車會慢十五分鐘。"),
    RegressionCase("delay_eta", "頭班車明天會晚半小時。", "頭班車明仔載會慢分半點鐘。"),
    RegressionCase("delay_eta", "今天的末班車提早十分鐘開車。", "今仔日的尾班車較早十分鐘開車。"),
    RegressionCase("delay_eta", "這班公車臨時改道，轉車時間也會較慢。", "這班公車臨時改道，轉車時間也會較慢。"),
    RegressionCase("delay_eta", "這班公車改道以後，可能不會經過縣政府。", "這班公車改道了後，可能袂經過縣政府。"),
    RegressionCase("delay_eta", "往縣政府的車今天改道。", "往縣政府的車今仔日改道。"),
    RegressionCase("delay_eta", "這班車現在先停駛，晚一點再公告。", "這班車這馬先停開，較慢閣公告。"),
    RegressionCase("delay_eta", "手機顯示不準，現場公告才準。", "手機顯示的無準，現場公告的才準。"),
    RegressionCase("delay_eta", "現在先照站牌公告，不要看手機時間。", "這馬先照站牌公告，莫看手機時間。"),
    RegressionCase("delay_eta", "回程車大概十分鐘後到。", "回程車差不多十分鐘後到。"),
    RegressionCase("delay_eta", "去火車站的車還有五分鐘。", "去火車頭的車閣有五分鐘。"),
    RegressionCase("delay_eta", "司機休息回來就會發車。", "司機歇睏轉來就會開車。"),
    RegressionCase("delay_eta", "司機會在下一站休息五分鐘，你不用下車。", "司機會佇下一站歇睏五分鐘，你免落車。"),
    # bus query time expressions / calendar
    RegressionCase("bus_time_queries", "請問下一班公車幾點到？", "借問後一班公車幾點到？"),
    RegressionCase("bus_time_queries", "這班公車多久一班？", "這班公車偌久一班？"),
    RegressionCase("bus_time_queries", "末班車星期幾比較早？", "尾班車禮拜幾較早？"),
    RegressionCase("bus_time_queries", "明天週幾有車？", "明仔載禮拜幾有車？"),
    RegressionCase("bus_time_queries", "哪一天有加班車？", "佗一工有加班車？"),
    RegressionCase("bus_time_queries", "請問下個月哪天開始改時刻表？", "借問後個月佗一工開始改時刻表？"),
    RegressionCase("bus_time_queries", "下個月月初時刻表會改嗎？", "後個月月頭時刻表會改無？"),
    RegressionCase("bus_time_queries", "這班車月底會停駛嗎？", "這班車月尾會停開無？"),
    RegressionCase("bus_time_queries", "從斗六到北港需要多少時間？", "對斗六到北港愛偌久？"),
    RegressionCase("bus_time_queries", "這條路線還要多長時間才會恢復？", "這條路線閣愛偌久才會恢復？"),
    RegressionCase("bus_time_queries", "首班車幾點發車？", "頭班車幾點開車？"),
    RegressionCase("bus_time_queries", "末班車幾點開？", "尾班車幾點開？"),
    RegressionCase("bus_time_queries", "平日多久一班？", "平日偌久一班？"),
    RegressionCase("bus_time_queries", "假日多久一班？", "假日偌久一班？"),
    RegressionCase("bus_time_queries", "尖峰時間多久一班？", "尖峰時間偌久一班？"),
    RegressionCase("bus_time_queries", "離峰時間多久一班？", "離峰時間偌久一班？"),
    RegressionCase("bus_time_queries", "這班車延誤多久？", "這班車延誤偌久？"),
    RegressionCase("bus_time_queries", "時刻表哪一天生效？", "時刻表佗一工生效？"),
    RegressionCase("bus_time_queries", "改點後第一班幾點開？", "改點後第一班幾點開？"),
    RegressionCase("bus_time_queries", "臨時班車幾點會到？", "臨時班車幾點會到？"),
    RegressionCase("bus_time_queries", "班距大概多久？", "班距差不多偌久？"),
    RegressionCase("bus_time_queries", "這條路線每小時一班。", "這條路線每一點鐘一班。"),
    RegressionCase("bus_time_queries", "平日每小時一班，假日每半小時一班。", "平日每一點鐘一班，假日每半點鐘一班。"),
    RegressionCase("bus_time_queries", "公車每多久一班？", "公車偌久一班？"),
    RegressionCase("bus_time_queries", "尖峰時間每幾分鐘一班？", "尖峰時間偌久一班？"),
    RegressionCase("bus_time_queries", "離峰時間每隔幾分鐘一班？", "離峰時間隔偌久一班？"),
    RegressionCase("bus_time_queries", "公車幾點收班？", "公車幾點收班？"),
    RegressionCase("bus_time_queries", "這條路線營運到幾點？", "這條路線營運到幾點？"),
    RegressionCase("bus_time_queries", "服務時間到幾點？", "服務時間到幾點？"),
    RegressionCase("bus_time_queries", "預計幾點到站？", "預計幾點到站？"),
    RegressionCase("bus_time_queries", "今天有營運嗎？", "今仔日有開無？"),
    RegressionCase("bus_time_queries", "明天正常營運嗎？", "明仔載正常開無？"),
    RegressionCase("bus_time_queries", "假日有行駛嗎？", "假日有開無？"),
    RegressionCase("bus_time_queries", "這條路線今天有營運嗎？", "這條路線今仔日有開無？"),
    RegressionCase("bus_time_queries", "颱風天有行駛嗎？", "風颱天有開無？"),
    RegressionCase("bus_time_queries", "春節期間正常營運嗎？", "春節期間正常開無？"),
    RegressionCase("bus_time_queries", "這班車今天照常行駛嗎？", "這班車今仔日照常開無？"),
    RegressionCase("bus_time_queries", "國定假日正常行駛嗎？", "國定假日正常開無？"),
    RegressionCase("bus_time_queries", "今天停駛還是正常營運？", "今仔日停開猶是正常開？"),
    RegressionCase("bus_time_queries", "發車時間是幾點？", "開車時間是幾點？"),
    RegressionCase("bus_time_queries", "到站時間是幾點？", "到站時間是幾點？"),
    RegressionCase("bus_time_queries", "末班時間是幾點？", "尾班時間是幾點？"),
    RegressionCase("bus_time_queries", "首班時間是幾點？", "頭班時間是幾點？"),
    RegressionCase("bus_time_queries", "末班發車時間是幾點？", "尾班開車時間是幾點？"),
    RegressionCase("bus_time_queries", "首班發車時間是幾點？", "頭班開車時間是幾點？"),
    RegressionCase("bus_time_queries", "下一班到站時間是幾點？", "後一班到站時間是幾點？"),
    RegressionCase("bus_time_queries", "末班大概什麼時候開？", "尾班差不多啥物時陣開？"),
    RegressionCase("bus_time_queries", "還要再等多久？", "猶愛閣等偌久？"),
    RegressionCase("bus_time_queries", "還要再等幾分鐘？", "猶愛閣等幾分鐘？"),
    RegressionCase("bus_time_queries", "大概要再等多久？", "差不多閣愛等偌久？"),
    RegressionCase("bus_time_queries", "大概要再等幾分鐘？", "差不多閣愛等幾分鐘？"),
    RegressionCase("bus_time_queries", "要再等多久才有車？", "愛閣等偌久才有車？"),
    RegressionCase("bus_time_queries", "要再等幾分鐘才有車？", "愛閣等幾分鐘才有車？"),
    RegressionCase("bus_time_queries", "從斗六到北港要多久？", "對斗六到北港要偌久？"),
    RegressionCase("bus_time_queries", "需要多久才會到？", "愛偌久才會到？"),
    RegressionCase("bus_time_queries", "大概需要多久？", "差不多愛偌久？"),
    RegressionCase("bus_time_queries", "還要多久才會到？", "猶愛偌久才會到？"),
    RegressionCase("bus_time_queries", "何時發車？", "啥物時陣開車？"),
    RegressionCase("bus_time_queries", "何時開始？", "啥物時陣開始？"),
    RegressionCase("bus_time_queries", "幾點以前要到？", "幾點以前愛到？"),
    RegressionCase("bus_time_queries", "多久以前要到？", "偌久以前愛到？"),
    RegressionCase("bus_time_queries", "最晚幾點要到？", "上晏幾點愛到？"),
    RegressionCase("bus_time_queries", "你何時要回來？", "你啥物時陣欲轉來？"),
    RegressionCase("bus_time_queries", "何時要出門？", "啥物時陣欲出門？"),
    RegressionCase("bus_time_queries", "多久之後會到？", "偌久後會到？"),
    RegressionCase("bus_time_queries", "多少分鐘後會到？", "幾分鐘後會到？"),
    RegressionCase("bus_time_queries", "每多少分鐘一班？", "偌久一班？"),
    RegressionCase("bus_time_queries", "每隔多少分鐘一班？", "隔偌久一班？"),
    RegressionCase("bus_time_queries", "每隔多少時間一班？", "隔偌久一班？"),
    RegressionCase("bus_time_queries", "下個月初會改嗎？", "後個月月頭會改無？"),
    RegressionCase("bus_time_queries", "週末幾點開始？", "禮拜尾幾點開始？"),
    RegressionCase("bus_time_queries", "下週末有車嗎？", "下禮拜尾有車無？"),
    RegressionCase("bus_time_queries", "前天幾點停駛？", "前日幾點停開？"),
    RegressionCase("bus_time_queries", "幾點之後才有車？", "幾點後才有車？"),
    RegressionCase("bus_time_queries", "幾點之前要到？", "幾點進前愛到？"),
    RegressionCase("bus_time_queries", "月底前要完成嗎？", "月尾進前要完成無？"),
    RegressionCase("bus_time_queries", "月底以前會公告嗎？", "月尾進前會公告無？"),
    RegressionCase("bus_time_queries", "這週末會加班嗎？", "這禮拜尾會加班無？"),
    RegressionCase("bus_time_queries", "今天早上幾點開始？", "今仔日早起幾點開始？"),
    RegressionCase("bus_time_queries", "今天上午幾點開始？", "今仔日早起幾點開始？"),
    RegressionCase("bus_time_queries", "今晚幾點結束？", "今仔暗幾點結束？"),
    RegressionCase("bus_time_queries", "上週末有車嗎？", "頂禮拜尾有車無？"),
    RegressionCase("bus_time_queries", "上周末有車嗎？", "頂禮拜尾有車無？"),
    RegressionCase("bus_time_queries", "這周末會加班嗎？", "這禮拜尾會加班無？"),
    RegressionCase("bus_time_queries", "下周末有車嗎？", "下禮拜尾有車無？"),
    RegressionCase("bus_time_queries", "幾點以後才有車？", "幾點後才有車？"),
    RegressionCase("bus_time_queries", "幾分鐘以後會到？", "幾分鐘後會到？"),
    RegressionCase("bus_time_queries", "多少小時後會到？", "幾點鐘後會到？"),
    RegressionCase("bus_time_queries", "要等到什麼時候？", "欲等到啥物時陣？"),
    RegressionCase("bus_time_queries", "等到什麼時候才有車？", "等到啥物時陣才有車？"),
    RegressionCase("bus_time_queries", "什麼時候以前要到？", "啥物時陣進前愛到？"),
    RegressionCase("bus_time_queries", "什麼時候之後才有車？", "啥物時陣後才有車？"),
    RegressionCase("bus_time_queries", "下個月底以前會公告嗎？", "後個月月尾進前會公告無？"),
    RegressionCase("bus_time_queries", "幾點之前要報到？", "幾點進前愛報到？"),
    RegressionCase("bus_time_queries", "多久以前要報到？", "偌久以前愛報到？"),
    RegressionCase("bus_time_queries", "本週幾點開始？", "這禮拜幾點開始？"),
    RegressionCase("bus_time_queries", "本周幾點開始？", "這禮拜幾點開始？"),
    RegressionCase("bus_time_queries", "本週末有車嗎？", "這禮拜尾有車無？"),
    RegressionCase("bus_time_queries", "本周末有車嗎？", "這禮拜尾有車無？"),
    RegressionCase("bus_time_queries", "本月初會改嗎？", "這个月月頭會改無？"),
    RegressionCase("bus_time_queries", "本月底以前會公告嗎？", "這个月月尾進前會公告無？"),
    RegressionCase("bus_time_queries", "本月底前要完成嗎？", "這个月月尾進前要完成無？"),
    RegressionCase("bus_time_queries", "上個月初有改嗎？", "頂個月月頭有改無？"),
    RegressionCase("bus_time_queries", "上個月底以前有公告嗎？", "頂個月月尾進前有公告無？"),
    RegressionCase("bus_time_queries", "一週內會通知嗎？", "一禮拜內會通知無？"),
    RegressionCase("bus_time_queries", "一周內會通知嗎？", "一禮拜內會通知無？"),
    RegressionCase("bus_time_queries", "兩週後會改嗎？", "兩禮拜後會改無？"),
    RegressionCase("bus_time_queries", "兩周後會改嗎？", "兩禮拜後會改無？"),
    RegressionCase("bus_time_queries", "凌晨幾點開始？", "半暝幾點開始？"),
    RegressionCase("bus_time_queries", "今天之內會公告嗎？", "今仔日內會公告無？"),
    RegressionCase("bus_time_queries", "幾點內要到？", "幾點進前愛到？"),
    RegressionCase("bus_time_queries", "每個月幾號公告？", "逐个月幾號公告？"),
    RegressionCase("bus_time_queries", "每月幾號公告？", "逐月幾號公告？"),
    RegressionCase("bus_time_queries", "每年幾月調整？", "逐年幾月調整？"),
    RegressionCase("bus_time_queries", "每隔多久一次？", "隔偌久一擺？"),
    RegressionCase("bus_time_queries", "多少天一次？", "幾工一擺？"),
    RegressionCase("bus_time_queries", "昨天晚上幾點停駛？", "昨暗幾點停開？"),
    RegressionCase("bus_time_queries", "前晚幾點停駛？", "前暗幾點停開？"),
    RegressionCase("bus_time_queries", "可以幫我查班距嗎？", "會當替我查班距無？"),
    RegressionCase("bus_time_queries", "可以幫我查首班車時間嗎？", "會當替我查頭班車時間無？"),
    RegressionCase("bus_time_queries", "可以幫我查公車營運時間嗎？", "會當替我查公車營運時間無？"),
    # payment / cards / ticketing
    RegressionCase("payment_cards", "如果你沒有零錢，可以先投現再去總站補票。", "若是你無零錢，會當先投現錢閣去總站補票。"),
    RegressionCase("payment_cards", "如果你的愛心卡刷不過，可以改用投現。", "若是你的愛心卡鑢袂過，會當改用投現錢。"),
    RegressionCase("payment_cards", "如果刷卡還是不過，我先幫你登記，再請你補票。", "若是鑢卡猶毋過，我先替你登記，再請你補票。"),
    RegressionCase("payment_cards", "如果刷卡還是失敗，就先投現金。", "若是刷袂過，就先投現錢。"),
    RegressionCase("payment_cards", "你先上車，補票到總站再處理。", "你先上車，補票到總站再處理。"),
    RegressionCase("payment_cards", "如果你要去總站補票，先跟司機說一聲。", "若是你欲去總站補票，先佮司機講一聲。"),
    RegressionCase("payment_cards", "這台刷卡機壞了，你到後門那台刷。", "這台刷卡機歹去矣，你到後門那台刷。"),
    RegressionCase("payment_cards", "愛心卡今天可以正常刷卡。", "愛心卡今仔日會當正常刷卡。"),
    RegressionCase("payment_cards", "老人卡感應不到的話，請你先跟司機說。", "若是老人卡感應袂著，請你先佮司機講。"),
    RegressionCase("payment_cards", "你沒有零錢的話，可以去便利商店換。", "若是你無零錢，會當去便利商店換。"),
    RegressionCase("payment_cards", "可以幫我查公車票價嗎？", "會當替我查公車票價無？"),
    RegressionCase("payment_cards", "可以幫我查票卡餘額嗎？", "會當替我查票卡餘額無？"),
    # accessibility / boarding
    RegressionCase("accessibility", "今天這班低底盤公車壞掉了，換成一般車。", "今仔日這班低底盤公車歹去矣，換成一般車。"),
    RegressionCase("accessibility", "這班車今天改成小車，所以沒有輪椅斜板。", "這班車今仔日改成小車，所以無輪椅斜板。"),
    RegressionCase("accessibility", "輪椅要上車的話，我先幫你放斜板。", "若是輪椅要上車，我先替你共斜板放落來。"),
    RegressionCase("accessibility", "你如果要推輪椅上車，等一下我先請大家讓一下。", "你若是要推輪椅上車，等咧我先請逐家讓一下。"),
    RegressionCase("accessibility", "你先不要排太前面，讓輪椅乘客先上車。", "你先莫排太頭前，讓坐輪椅的乘客先上車。"),
    RegressionCase("accessibility", "嬰兒車也可以上車，但請先收好。", "嬰仔車也會當上車，但請先收予好。"),
    RegressionCase("accessibility", "這班低底盤公車今天沒有來。", "這班低底盤公車今仔日無來。"),
    RegressionCase("accessibility", "這班車今天不載腳踏車。", "這班車今仔日無載跤踏車。"),
    # route / transfer / destinations
    RegressionCase("route_transfer", "這班車今天不會再往前開，你要在火車站轉車。", "這班車今仔日袂閣往前開，你欲在火車頭轉車。"),
    RegressionCase("route_transfer", "這班公車會先到高鐵站，再回到斗六火車站。", "這班公車會先到高鐵站，閣轉到斗六火車站。"),
    RegressionCase("route_transfer", "這班車先到總站，再開去高鐵站。", "這班車先到總站，再開去高鐵站。"),
    RegressionCase("route_transfer", "這班車等一下會先進醫院，再回到車站。", "這班車等咧會先進病院，閣轉到車站。"),
    RegressionCase("route_transfer", "這班車等一下會先繞去市場，再回到火車站。", "這班車等咧會先踅去市場，閣轉到火車頭。"),
    RegressionCase("route_transfer", "如果你要轉火車，這班車可能來不及。", "若是你欲轉火車，這班車可能袂赴。"),
    RegressionCase("route_transfer", "這班車到總站就不開了。", "這班車到總站就不開矣。"),
    RegressionCase("route_transfer", "這班車不會進醫院急診門口。", "這班車袂進病院急診門跤口。"),
    RegressionCase("route_transfer", "你要去門診的話，在病院門口下就可以了。", "若是你欲去門診，佇病院門跤口下就會當矣。"),
    RegressionCase("route_transfer", "如果你要去學校裡面，要在校門口下車再走進去。", "若是你欲去學校內底，要在校門跤口落車才行進去。"),
    RegressionCase("route_transfer", "要去老街的話，你在外圍下車再走進去。", "若是欲去老街，你佇外圍落車才行進去。"),
    RegressionCase("route_transfer", "要去朝天宮的話，你在外圍下車就可以。", "若是欲去朝天宮，你佇外圍落車就會當。"),
    RegressionCase("route_transfer", "你去對面坐回火車站那班。", "你去對面坐回火車頭彼班。"),
    RegressionCase("route_transfer", "往市場的車在這裡排隊。", "欲去市場的車佇遮排線。"),
    RegressionCase("route_transfer", "我想改搭公車。", "我想欲改搭公車。"),
    RegressionCase("route_transfer", "我想改轉車地點。", "我想欲改轉車地點。"),
    RegressionCase("route_transfer", "可以幫我查回程班次嗎？", "會當替我查回程班次無？"),
    # Yunlin station names / attractions
    RegressionCase("yunlin_stops_attractions", "這班車會到雲林布袋戲館嗎？", "這班車會到雲林布袋戲館無？"),
    RegressionCase("yunlin_stops_attractions", "北港朝天宮站牌在水道頭文化園區旁邊嗎？", "北港朝天宮站牌佇水道頭文化園區隔壁無？"),
    RegressionCase("yunlin_stops_attractions", "我要從高鐵雲林站去虎尾糖廠。", "我欲對高鐵雲林站去虎尾糖廠。"),
    RegressionCase("yunlin_stops_attractions", "這班公車有停西螺延平老街嗎？", "這班公車有停西螺延平老街無？"),
    RegressionCase("yunlin_stops_attractions", "口湖遊客中心到成龍濕地要轉車嗎？", "口湖遊客中心到成龍濕地愛轉車無？"),
    RegressionCase("yunlin_stops_attractions", "虎尾驛、虎尾鐵橋和合同廳舍都在虎尾嗎？", "虎尾驛、虎尾鐵橋和合同廳舍攏佇虎尾無？"),
    RegressionCase("yunlin_stops_attractions", "西螺福興宮和丸莊醬油觀光工廠附近有站牌嗎？", "西螺福興宮和丸莊醬油觀光工廠附近有站牌無？"),
    RegressionCase("yunlin_stops_attractions", "台灣好行會停麥寮拱範宮和三條崙海清宮。", "台灣好行會停麥寮拱範宮和三條崙海清宮。"),
    RegressionCase("yunlin_stops_attractions", "站牌在高鐵雲林站旁邊嗎？", "站牌佇高鐵雲林站隔壁無？"),
    RegressionCase("yunlin_shuttle_routes", "斗六古坑線會停雲中街和社口遊客中心嗎？", "斗六古坑線會停雲中街和社口遊客中心無？"),
    RegressionCase("yunlin_shuttle_routes", "綠色隧道到古坑嘉興宮要多久？", "綠色隧道到古坑嘉興宮要偌久？"),
    RegressionCase("yunlin_shuttle_routes", "福祿壽酒廠和永光故事屋都在古坑嗎？", "福祿壽酒廠和永光故事屋攏佇古坑無？"),
    RegressionCase("yunlin_shuttle_routes", "劍湖山世界到華山咖啡大街有接駁車嗎？", "劍湖山世界到華山咖啡大街有接駁車無？"),
    RegressionCase("yunlin_shuttle_routes", "雲西線從北港武德宮開到高鐵嘉義站。", "雲西線對北港武德宮開到高鐵嘉義站。"),
    RegressionCase("yunlin_shuttle_routes", "北港春生活博物館、黃金蝙蝠生態館和戰水鯨湖廣場都在雲西線嗎？", "北港春生活博物館、黃金蝙蝠生態館和戰水鯨湖廣場攏佇雲西線無？"),
    RegressionCase("yunlin_shuttle_routes", "顏厝寮聚落和北港1911好庫文化產業園區附近有站牌嗎？", "顏厝寮聚落和北港1911好庫文化產業園區附近有站牌無？"),
    RegressionCase("yunlin_shuttle_routes", "草嶺線會經過鎮西國小、水岸藝術公園和成大醫院嗎？", "草嶺線會經過鎮西國小、水岸藝術公園和成大醫院無？"),
    RegressionCase("yunlin_shuttle_routes", "荷苞山桐花公園到草嶺公園還要多久？", "荷苞山桐花公園到草嶺公園猶愛偌久？"),
    RegressionCase("yunlin_shuttle_routes", "新草嶺國小站牌在東𤧥山莊前面嗎？", "新草嶺國小站牌佇東𤧥山莊頭前無？"),
    RegressionCase("yunlin_tourism_attractions", "土庫順天宮、土庫庄役場和源順芝麻觀光油廠都在土庫嗎？", "土庫順天宮、土庫庄役場和源順芝麻觀光油廠攏佇土庫無？"),
    RegressionCase("yunlin_tourism_attractions", "圖南咖啡故事館、行啟記念館和三小市集都在斗六嗎？", "圖南咖啡故事館、行啟記念館和三小市集攏佇斗六無？"),
    RegressionCase("yunlin_tourism_attractions", "石榴車站附近有榴中社區、新德豐碾米廠和張氏宗祠嗎？", "石榴車站附近有榴中社區、新德豐米絞和張氏宗祠無？"),
    RegressionCase("yunlin_tourism_attractions", "蘿莎玫瑰莊園到劍湖山世界幸福摩天輪要轉車嗎？", "蘿莎玫瑰莊園到劍湖山世界幸福摩天輪愛轉車無？"),
    RegressionCase("yunlin_tourism_attractions", "華山文學步道、幽情谷步道和水濂洞哪一站比較近？", "華山文學步道、幽情谷步道和水濂洞佗一站較近？"),
    RegressionCase("yunlin_tourism_attractions", "峭壁雄風步道、小天梯、雲嶺之丘和五元二角都在古坑嗎？", "峭壁雄風步道、小天梯、雲嶺之丘和五元二角攏佇古坑無？"),
    RegressionCase("yunlin_tourism_attractions", "北港義民廟、北港工藝坊和北港春生活博物館附近有站牌嗎？", "北港義民廟、北港工藝坊和北港春生活博物館附近有站牌無？"),
    RegressionCase("yunlin_tourism_attractions", "馬蹄蛤主題館、台灣鯛生態創意園區和沐藝堂都在口湖嗎？", "馬蹄蛤主題館、台灣鯛生態創意園區和沐藝堂攏佇口湖無？"),
    RegressionCase("yunlin_tourism_attractions", "椬梧滯洪池到好蝦冏男社和成龍濕地要多久？", "椬梧滯洪池到好蝦冏男社和成龍濕地要偌久？"),
    RegressionCase("yunlin_coastal_caoling", "斗南他里霧文化園區和斗南圓環附近有公車嗎？", "斗南他里霧文化園區和斗南圓環附近有公車無？"),
    RegressionCase("yunlin_coastal_caoling", "北港限定早餐、北港巷弄美食和北港美食小吃都在北港嗎？", "北港限定早餐、北港巷弄美食和北港美食小吃攏佇北港無？"),
    RegressionCase("yunlin_coastal_caoling", "水道頭文化園區遊客中心到北港女兒橋要走多久？", "水道頭文化園區遊客中心到北港女兒橋要走偌久？"),
    RegressionCase("yunlin_coastal_caoling", "椬梧滯洪池、成龍濕地和金湖沙灘都在雲西線嗎？", "椬梧滯洪池、成龍濕地和金湖沙灘攏佇雲西線無？"),
    RegressionCase("yunlin_coastal_caoling", "四湖參天宮、三條崙海水浴場和箔子寮漁港有公車嗎？", "四湖參天宮、三條崙海水浴場和箔子寮漁港有公車無？"),
    RegressionCase("yunlin_coastal_caoling", "草嶺風景區、石壁風景區和草嶺古道在哪裡下車？", "草嶺風景區、石壁風景區和草嶺古道佇佗位落車？"),
    RegressionCase("yunlin_coastal_caoling", "大飛山、杉林步道和遊龍湖步道今天有接駁車嗎？", "大飛山、杉林步道和遊龍湖步道今仔日有接駁車無？"),
    RegressionCase("yunlin_coastal_caoling", "雲林記憶Cool和雲林故事館都在虎尾市區嗎？", "雲林記憶Cool和雲林故事館攏佇虎尾市區無？"),
    RegressionCase("yunlin_coastal_caoling", "虎尾建國一村和北溪剪紙藝術村附近有站牌嗎？", "虎尾建國一村和北溪剪紙藝術村附近有站牌無？"),
    RegressionCase("yunlin_caoling_minor_stops", "早寮、二坪仔和東內寮附近有站牌嗎？", "早寮、二坪仔和東內寮附近有站牌無？"),
    RegressionCase("yunlin_caoling_minor_stops", "小旗仔、檳榔宅、外湖和內湖都在草嶺線嗎？", "小旗仔、檳榔宅、外湖和內湖攏佇草嶺線無？"),
    RegressionCase("yunlin_caoling_minor_stops", "草嶺線會經過東和、受天宮和環球科技大學側門嗎？", "草嶺線會經過東和、受天宮和環球科技大學側門無？"),
    RegressionCase("yunlin_caoling_minor_stops", "站牌在檳榔宅附近嗎？", "站牌佇檳榔宅附近無？"),
    RegressionCase("yunlin_caoling_minor_stops", "外湖到內湖要走多久？", "外湖到內湖要走偌久？"),
    RegressionCase("yunlin_caoling_minor_stops", "新草嶺國小到草嶺山莊會先經過草嶺嗎？", "新草嶺國小到草嶺山莊會先過草嶺無？"),
    RegressionCase("yunlin_douliu_xiluo_attractions", "太平老街和聖玫瑰天主堂附近有公車嗎？", "太平老街和聖玫瑰天主堂附近有公車無？"),
    RegressionCase("yunlin_douliu_xiluo_attractions", "西螺廣福宮、西螺福興宮和西螺大橋都在高鐵雲林站北邊嗎？", "西螺廣福宮、西螺福興宮和西螺大橋攏佇高鐵雲林站北爿無？"),
    RegressionCase("yunlin_douliu_xiluo_attractions", "斗六棒球場、雲中街和黑膠音樂故事館要在哪裡下車？", "斗六野球場、雲中街和黑膠音樂故事館要佇佗位落車？"),
    RegressionCase("yunlin_douliu_xiluo_attractions", "凹凸咖啡、猿樂作和貝歐克尼Balcony乾燥花都在雲中街嗎？", "凹凸咖啡、猿樂作和貝歐克尼Balcony焦燥花攏佇雲中街無？"),
    RegressionCase("yunlin_douliu_xiluo_attractions", "Mr. Lobby Coffee Roaster和劍湖山世界樂園有台灣好行優惠嗎？", "Mr. Lobby Coffee Roaster和劍湖山世界樂園有台灣好行優惠無？"),
    RegressionCase("yunlin_douliu_xiluo_attractions", "站牌在斗六棒球場旁邊嗎？", "站牌佇斗六野球場隔壁無？"),
    RegressionCase("yunlin_douliu_xiluo_attractions", "西螺媽廟到西螺大橋要轉車嗎？", "西螺媽廟到西螺大橋愛轉車無？"),
    RegressionCase("yunlin_shuttle_food_stops", "蜜蜂故事館和華山咖啡大街物產館都在斗六古坑線嗎？", "蜜蜂故事館和華山咖啡大街物產館攏佇斗六古坑線無？"),
    RegressionCase("yunlin_shuttle_food_stops", "鄉村休閒農莊、咖啡大街民宿和莿桐花咖啡附近有站牌嗎？", "鄉村休閒農莊、咖啡大街民宿和莿桐花咖啡附近有站牌無？"),
    RegressionCase("yunlin_shuttle_food_stops", "石墩庭園咖啡在華山咖啡大街附近嗎？", "石墩庭園咖啡佇華山咖啡大街附近無？"),
    RegressionCase("yunlin_shuttle_food_stops", "蝦の故鄉興義軒休閒園區和北港當歸鴨都在雲西線嗎？", "蝦の故鄉興義軒休閒園區和北港當歸鴨攏佇雲西線無？"),
    RegressionCase("yunlin_shuttle_food_stops", "站牌在蜜蜂故事館前面嗎？", "站牌佇蜜蜂故事館頭前無？"),
    RegressionCase("yunlin_shuttle_food_stops", "站牌在蝦の故鄉 興義軒休閒園區前面嗎？", "站牌佇蝦の故鄉興義軒休閒園區頭前無？"),
    RegressionCase("yunlin_shuttle_food_stops", "阿甘薯叔 雲林高鐵門市在高鐵雲林站裡面嗎？", "阿甘薯叔雲林高鐵門市佇高鐵雲林站內底無？"),
    # destination clarification prompts
    RegressionCase("destination_prompt", "臺北範圍很廣，請問您具體要去哪個地點呢。", "台北真大，借問你欲去佗位？"),
    RegressionCase("destination_prompt", "請問您具體要去哪個地點呢。", "借問你欲去佗位？"),
    RegressionCase("destination_prompt", "您好，請問您要去臺北的哪個地方呢。", "你好，借問你欲去台北佗位？"),
    RegressionCase("destination_prompt", "只要告訴我目的地，我就能為您查詢最近的公車路線", "只要共我講你欲去佗位，我就會當替你查較近的公車路線"),
    RegressionCase("destination_prompt", "您好，請問您要去臺北的哪個地方呢。只要告訴我目的地，我就能為您查詢最近的公車路線", "你好，借問你欲去台北佗位？只要共我講你欲去佗位，我就會當替你查較近的公車路線"),
    RegressionCase("destination_prompt", "您好，請問您是指新莊區嗎。", "你好，借問你是指新莊區無？"),
    RegressionCase("destination_prompt", "請告訴我您想前往的新莊具體地點，我才能為您查詢", "請共我講你欲去新莊佗位，我才會當替你查"),
    RegressionCase("destination_prompt", "您好，請問您是指新莊區嗎。請告訴我您想前往的新莊具體地點，我才能為您查詢", "你好，借問你是指新莊區無？請共我講你欲去新莊佗位，我才會當替你查"),
    RegressionCase("destination_prompt", "您要去的是獅崙嗎。", "你欲去的是獅崙無？"),
    RegressionCase("destination_prompt", "請告訴我您想前往獅崙的哪個地點", "請共我講你欲去獅崙佗位"),
    RegressionCase("destination_prompt", "您要去的是獅崙嗎。請告訴我您想前往獅崙的哪個地點", "你欲去的是獅崙無？請共我講你欲去獅崙佗位"),
    # station service / lost property / redirect
    RegressionCase("service_redirect", "這個問題要問承辦單位，我這邊只能查公車班次。", "這个問題要問承辦單位，我遮干焦會當查公車班次。"),
    RegressionCase("service_redirect", "這個問題跟公車無關，我沒辦法回答。", "這个問題佮公車無關，我無法度回答。"),
    RegressionCase("service_redirect", "這不是我們站務可以決定的，你要問總站。", "這毋是阮站務會當決定的，你欲問總站。"),
    RegressionCase("service_redirect", "這個不是公車問題，請去問警察。", "這个毋是公車問題，請去問警察。"),
    RegressionCase("service_redirect", "如果你只是問廁所在哪裡，我可以跟你說。", "若是你只是問便所佇佗位，我會當共你講。"),
    RegressionCase("service_redirect", "這裡有公車動態可以查。", "遮有公車動態會當查。"),
    RegressionCase("service_redirect", "如果你只是要知道公車到哪裡，我可以幫你查動態。", "若是你只是欲知影公車到佗位，我會當替你查動態。"),
    RegressionCase("service_redirect", "如果你要查公車到哪裡，我可以幫你看。", "若是你欲查公車到佗位，我會當替你看。"),
    RegressionCase("service_redirect", "如果你要找失物，我可以幫你轉給總站處理。", "若是你欲找失物，我會當替你轉去總站處理。"),
    RegressionCase("service_redirect", "如果你要查失物，我可以先幫你記下車牌和時間。", "若是你欲查失物，我會當先幫你記落車牌佮時間。"),
    RegressionCase("service_redirect", "如果你要找失物，我先幫你記車牌。", "若是你欲找失物，我先替你記車牌。"),
    RegressionCase("service_redirect", "失物要送回總站，你下午再打電話確認。", "遺失物愛送轉去總站，你下晝閣敲電話確認。"),
    RegressionCase("service_redirect", "可以幫我查公車路線嗎？", "會當替我查公車路線無？"),
    RegressionCase("service_redirect", "可以幫我查路線變更嗎？", "會當替我查路線變更無？"),
    RegressionCase("service_redirect", "可以幫我查公車站名嗎？", "會當替我查公車站名無？"),
    RegressionCase("service_redirect", "可以幫我查乘車規定嗎？", "會當替我查乘車規定無？"),
    RegressionCase("service_redirect", "可以幫我查公車時刻表嗎？", "會當替我查公車時刻表無？"),
    RegressionCase("service_redirect", "可以幫我查公車到站時間嗎？", "會當替我查公車到站時間無？"),
    RegressionCase("service_redirect", "可以幫我查公車路況嗎？", "會當替我查公車路況無？"),
    RegressionCase("service_redirect", "可以幫我查公車班次嗎？", "會當替我查公車班次無？"),
    RegressionCase("service_redirect", "我需要查公車班次。", "我欲查公車班次。"),
    RegressionCase("service_redirect", "我需要確認公車班次。", "我欲確認公車班次。"),
    RegressionCase("service_redirect", "我想確認公車班次。", "我想欲確認公車班次。"),
    RegressionCase("service_redirect", "可以幫我確認公車班次嗎？", "會當替我確認公車班次無？"),
    RegressionCase("service_redirect", "請幫我確認公車班次。", "請替我確認公車班次。"),
    RegressionCase("service_redirect", "麻煩你幫我確認公車班次。", "麻煩你替我確認公車班次。"),
    RegressionCase("service_redirect", "幫我確認公車班次。", "替我確認公車班次。"),
    # weather / crowd / queue
    RegressionCase("weather_crowd", "如果車子太滿，站牌這邊就先不要再排。", "若是車子太滿，站牌遮就先莫再排。"),
    RegressionCase("weather_crowd", "等一下如果車子太滿，司機可能不會讓你上車。", "等咧若是車子太滿，司機可能袂予你上車。"),
    RegressionCase("weather_crowd", "站牌這邊現在人很多，你先到旁邊等比較安全。", "站牌遮這馬人誠濟，你先到邊仔等較安全。"),
    RegressionCase("weather_crowd", "連假人很多，你先排旁邊一點。", "連假人誠濟，你先排較邊仔。"),
    RegressionCase("weather_crowd", "今天雨比較大，你先到騎樓下等，車到我再叫你。", "今仔日雨較大，你先到亭仔跤等，車到我才叫你。"),
    RegressionCase("weather_crowd", "你先到亭仔跤等，雨停了再過來。", "你先到亭仔跤等，雨停了再過來。"),
    RegressionCase("weather_crowd", "今天雨很大，車班可能不穩定。", "今仔日雨真大，車班可能不穩定。"),
    RegressionCase("weather_crowd", "廟口前面今天不能停車。", "廟埕頭前今仔日袂當停車。"),
    # misc route / wording
    RegressionCase("misc", "這班車已經客滿了，麻煩你等下一班。", "這班車已經客滿矣，麻煩你等後一班。"),
    RegressionCase("misc", "你先不要上車，讓老人先上。", "你先莫上車，讓老人先上。"),
    RegressionCase("misc", "回程車大概十分鐘後到。", "回程車差不多十分鐘後到。"),
    RegressionCase("misc", "站牌旁邊那台機器可以查時刻。", "站牌邊仔那台機器會當查時刻。"),
    RegressionCase("misc", "今天站牌移到巷口那邊。", "今仔日站牌移到巷口彼爿。"),
    RegressionCase("misc", "行李太大件的話，麻煩你放旁邊。", "若是行李太大件，麻煩你囥邊仔。"),
    RegressionCase("misc", "這班公車今天不停靠北港朝天宮。", "這班公車今仔日無停北港朝天宮。"),
    RegressionCase("misc", "可以幫我查站牌位置嗎？", "會當替我查站牌位置無？"),
    RegressionCase("misc", "可以幫我查站牌編號嗎？", "會當替我查站牌編號無？"),
]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="公車站務情境 regression runner")
    parser.add_argument("--rounds", type=int, default=1, help="重複跑幾輪 exact-match 回歸")
    parser.add_argument(
        "--category",
        action="append",
        default=[],
        help="只跑指定 category，可重複傳入",
    )
    parser.add_argument("--list-categories", action="store_true", help="列出所有 category")
    parser.add_argument("--show-pass", action="store_true", help="顯示每筆通過案例")
    parser.add_argument("--fail-fast", action="store_true", help="遇到第一個 mismatch 就停止")
    return parser


def _selected_cases(categories: list[str]) -> list[RegressionCase]:
    if not categories:
        return BUS_REGRESSION_CASES
    wanted = set(categories)
    return [case for case in BUS_REGRESSION_CASES if case.category in wanted]


def main() -> int:
    args = _build_parser().parse_args()
    categories = sorted({case.category for case in BUS_REGRESSION_CASES})
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

    print(
        {
            "rounds": args.rounds,
            "case_count": len(cases),
            "categories": dict(sorted(category_counts.items())),
        }
    )

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
