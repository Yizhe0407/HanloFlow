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


CONVERSATION_REGRESSION_CASES: list[RegressionCase] = [
    # greetings / farewells
    RegressionCase("greetings", "收到。", "收著。"),
    RegressionCase("greetings", "了解。", "知影。"),
    RegressionCase("greetings", "好久不見。", "好久無見。"),
    RegressionCase("greetings", "掰掰。", "再會。"),
    RegressionCase("greetings", "我回來了。", "我轉來矣。"),
    RegressionCase("greetings", "先走了，再見。", "行先矣，再會。"),
    # status check / location
    RegressionCase("status_check", "到哪了？", "到佗位矣？"),
    RegressionCase("status_check", "剛到家。", "拄到厝。"),
    RegressionCase("status_check", "出發了沒？", "出發矣未？"),
    RegressionCase("status_check", "你到了嗎？", "你到矣無？"),
    RegressionCase("status_check", "我快到了。", "我欲到矣。"),
    RegressionCase("status_check", "下班了沒？", "下班未？"),
    RegressionCase("status_check", "你今天幾點下班？", "你今仔日幾點下班？"),
    # daily chat
    RegressionCase("daily_chat", "你吃飯了嗎？", "你食飯矣無？"),
    RegressionCase("daily_chat", "你吃飽了嗎？", "你食飽未？"),
    RegressionCase("daily_chat", "你最近怎麼樣？", "你最近啥款？"),
    RegressionCase("daily_chat", "好久不見，你最近怎麼樣？", "好久無見，你最近啥款？"),
    RegressionCase("daily_chat", "我現在有點忙。", "我這馬有淡薄仔忙。"),
    RegressionCase("daily_chat", "我有點累了，先去休息一下。", "我有淡薄仔累矣，先去歇睏一下。"),
    # common responses
    RegressionCase("daily_response", "沒事。", "無代誌。"),
    RegressionCase("daily_response", "辛苦了。", "辛苦你矣。"),
    RegressionCase("daily_response", "你辛苦了。", "你辛苦矣。"),
    RegressionCase("daily_response", "麻煩你了。", "麻煩你。"),
    RegressionCase("daily_response", "好，我知道了。", "好，我知影矣。"),
    RegressionCase("daily_response", "沒問題。", "無問題。"),
    RegressionCase("daily_response", "還給你。", "還給你。"),
    RegressionCase("daily_response", "我都願意傾聽喔", "我攏肯用心聽你講喔"),
    RegressionCase("daily_response", "我都肯傾聽喔", "我攏肯聽你講喔"),
    RegressionCase("daily_response", "我都願意傾聽", "我攏肯用心聽你講"),
    RegressionCase("daily_response", "我都肯傾聽", "我攏肯聽你講"),
    RegressionCase("daily_response", "才忍不住想說給聽的", "才忍袂牢想講予聽的"),
    RegressionCase("daily_response", "才忍不住想講給聽的", "才忍袂牢想講予聽的"),
    RegressionCase("daily_response", "台北和台中都有不少不錯的選擇", "台北佮台中攏有袂少袂䆀的選擇"),
    RegressionCase("daily_response", "聽到你這樣說", "聽著你按呢講"),
    RegressionCase("daily_response", "你方便說話嗎？", "你方便講話無？"),
    # news / summit
    RegressionCase("news_summit", "高峰會", "峰會"),
    RegressionCase("news_summit", "民主高峰會", "民主峰會"),
    RegressionCase("news_summit", "哥本哈根民主高峰會", "哥本哈根民主峰會"),
    RegressionCase("news_summit", "國際高峰會", "國際峰會"),
    RegressionCase("news_summit", "氣候高峰會", "氣候峰會"),
    RegressionCase("news_summit", "經濟高峰會", "經濟峰會"),
    RegressionCase("news_summit", "領袖高峰會", "領袖峰會"),
    RegressionCase("news_summit", "跨國高峰會", "跨國峰會"),
    RegressionCase("news_summit", "政策高峰會", "政策峰會"),
    RegressionCase("news_summit", "世界高峰會", "世界峰會"),
    # news / welfare and childcare
    RegressionCase("news_welfare", "情緒管控", "情緒控制"),
    RegressionCase("news_welfare", "情緒管控不當", "情緒控制不當"),
    RegressionCase("news_welfare", "情緒失控", "情緒失控"),
    RegressionCase("news_welfare", "情緒起伏不定", "情緒起伏不定"),
    RegressionCase("news_welfare", "綜合所得稅", "綜合所得稅"),
    RegressionCase("news_welfare", "所得級距", "所得級距"),
    RegressionCase("news_welfare", "健保補助", "健保補助"),
    RegressionCase("news_welfare", "遺產稅扣除額", "遺產稅扣除額"),
    RegressionCase("news_welfare", "教保服務機構", "教保服務機構"),
    RegressionCase("news_welfare", "文化內容策進院", "文化內容策進院"),
    RegressionCase("news_welfare", "家戶綜所稅率5%以下。", "家戶綜所稅率百分之五以下。"),
    RegressionCase("news_welfare", "上升5%。", "上升百分之五。"),
    RegressionCase("news_welfare", "100%達成。", "百分之百達成。"),
    # news / CTS
    RegressionCase("news_cts_terms", "地面師", "地面師"),
    RegressionCase("news_cts_terms", "台版地面師", "台版地面師"),
    RegressionCase("news_cts_terms", "台版地面師詐騙", "台版地面師詐騙"),
    RegressionCase("news_cts_terms", "高壓線", "高壓線"),
    RegressionCase("news_cts_terms", "勾斷高壓線", "勾斷高壓線"),
    RegressionCase("news_cts_terms", "堤防", "堤防"),
    RegressionCase("news_cts_terms", "堤防工程", "堤防工程"),
    RegressionCase("news_cts_terms", "堤防工程怪手勾斷高壓線", "堤防工程怪手勾斷高壓線"),
    RegressionCase("news_cts_terms", "廠商代刀", "廠商代刀"),
    RegressionCase("news_cts_terms", "甲級動員", "甲級動員"),
    RegressionCase("news_cts_terms", "故宮博物院", "故宮博物院"),
    RegressionCase("news_cts_terms", "故宮博物院政務副院長", "故宮博物院政務副院長"),
    RegressionCase("news_cts_terms", "故宮博物院政務副院長黃永泰", "故宮博物院政務副院長黃永泰"),
    RegressionCase("news_cts_terms", "經濟部政務次長", "經濟部政務次長"),
    RegressionCase("news_cts_terms", "經濟部政務次長何晉滄", "經濟部政務次長何晉滄"),
    RegressionCase("news_cts_terms", "文化內容策進院", "文化內容策進院"),
    RegressionCase("news_cts_terms", "文策院董事之一", "文策院董事之一"),
    RegressionCase("news_cts_terms", "文化內容策進院的董事", "文化內容策進院的董事"),
    RegressionCase("news_cts_terms", "文策院也證實了", "文策院也證實了"),
    RegressionCase("news_cts_terms", "新任董事之一", "新任董事之一"),
    RegressionCase("news_cts_terms", "提案大會設立", "提案大會設立"),
    RegressionCase("news_cts_terms", "在文策院提案大會設立", "在文策院提案大會設立"),
    RegressionCase("news_cts_terms", "迎接新的身分", "迎接新的身分"),
    RegressionCase("news_cts_terms", "台灣主權獨立的國家", "台灣主權獨立的國家"),
    RegressionCase("news_cts_terms", "參與國際社會的決心", "參與國際社會的決心"),
    RegressionCase("news_cts_terms", "林志玲跨界接文策院董事", "林志玲跨界接文策院董事"),
    RegressionCase("news_cts_terms", "林志玲學經歷", "林志玲學經歷"),
    RegressionCase("news_cts_terms", "文化及影視推廣", "文化及影視推廣"),
    RegressionCase("news_cts_terms", "林志玲未來力量獎", "林志玲未來力量獎"),
    RegressionCase("news_cts_terms", "國際合製計畫", "國際合製計畫"),
    RegressionCase("news_cts_terms", "VR沉浸式內容開發", "VR沉浸式內容開發"),
    RegressionCase("news_cts_terms", "文化領域以及公益活動", "文化領域以及公益活動"),
    RegressionCase("news_cts_terms", "投身文化領域以及公益活動", "投身文化領域以及公益活動"),
    RegressionCase("news_cts_terms", "林志玲未來力量獎希望鼓勵富有創意和遠見的說故事者", "林志玲未來力量獎希望鼓勵富有創意和遠見的說故事者"),
    RegressionCase("news_cts_terms", "富有創意和遠見的說故事者", "富有創意和遠見的說故事者"),
    RegressionCase("news_cts_terms", "藝人林志玲將接任文策院董事", "藝人林志玲將接任文策院董事"),
    RegressionCase("news_cts_terms", "台灣傑出的影視人才被世界看見", "台灣傑出的影視人才被世界看見"),
    RegressionCase("news_cts_terms", "文化及影視推廣帶來不同的能量", "文化及影視推廣帶來不同的能量"),
    RegressionCase("news_cts_terms", "台灣文化及影視推廣帶來不同的能量", "台灣文化及影視推廣帶來不同的能量"),
    RegressionCase("news_cts_terms", "台灣文化及影視推廣注入一股新的力量", "台灣文化及影視推廣注入一股新的力量"),
    RegressionCase("news_cts_terms", "藝人林志玲去2025年才以個人名義", "藝人林志玲去二千零二十五年才以個人名義"),
    RegressionCase("news_cts_terms", "去2025年才以個人名義", "去二千零二十五年才以個人名義"),
    RegressionCase("news_cts_terms", "女性影視工作者", "女性影視工作者"),
    RegressionCase("news_cts_terms", "透過電影的力量", "透過電影的力量"),
    RegressionCase("news_cts_terms", "文化領域的滿滿熱愛", "文化領域的滿滿熱愛"),
    RegressionCase("news_cts_terms", "電視節目與戲劇製作", "電視節目與戲劇製作"),
    RegressionCase("news_cts_terms", "文化意含", "文化意含"),
    RegressionCase("news_cts_terms", "深耕不同領域", "深耕不同領域"),
    RegressionCase("news_cts_terms", "台灣文化推廣帶來不同的能量", "台灣文化推廣帶來不同的能量"),
    RegressionCase("news_cts_terms", "看出她對文化領域的滿滿熱愛", "看出她對文化領域的滿滿熱愛"),
    RegressionCase("news_cts_terms", "為台灣文化推廣帶來不同的能量", "為台灣文化推廣帶來不同的能量"),
    RegressionCase("news_cts_terms", "讓台灣傑出的影視人才被世界看見", "讓台灣傑出的影視人才被世界看見"),
    RegressionCase("news_cts_terms", "多倫多大學", "多倫多大學"),
    RegressionCase("news_cts_terms", "加拿大多倫多大學經濟美術學系雙主修畢業", "加拿大多倫多大學經濟美術學系雙主修畢業"),
    RegressionCase("news_cts_terms", "熊鷹研究團隊", "熊鷹研究團隊"),
    RegressionCase("news_cts_terms", "生態紀錄片導演", "生態紀錄片導演"),
    RegressionCase("news_cts_terms", "原住民部落", "原住民部落"),
    RegressionCase("news_cts_terms", "原住民部落的文化脈絡", "原住民部落的文化脈絡"),
    RegressionCase("news_cts_terms", "熊鷹羽毛利用的文化", "熊鷹羽毛利用的文化"),
    RegressionCase("news_cts_terms", "熊鷹羽毛對原住民來說也具有特殊的文化意涵", "熊鷹羽毛對原住民來說也具有特殊的文化意涵"),
    RegressionCase("news_cts_terms", "生態和鳥類的議題", "生態和鳥類的議題"),
    RegressionCase("news_cts_terms", "文化保存與生態保育之間", "文化保存與生態保育之間"),
    RegressionCase("news_cts_terms", "文化保存與生態保育之間取得平衡", "文化保存與生態保育之間取得平衡"),
    RegressionCase("news_cts_terms", "在文化保存與生態保育之間取得平衡", "在文化保存與生態保育之間取得平衡"),
    RegressionCase("news_cts_terms", "國際合製計畫作品遍及柏林釜山各大頂尖影展", "國際合製計畫作品遍及柏林釜山各大頂尖影展"),
    RegressionCase("news_cts_terms", "報稅扶養岳父大人", "報稅扶養岳父大人"),
    RegressionCase("news_cts_terms", "報稅扶養岳父大人笑不出來", "報稅扶養岳父大人笑不出來"),
    RegressionCase("news_cts_terms", "報稅扶養前得要掐指算一算", "報稅扶養前得要掐指算一算"),
    RegressionCase("news_cts_terms", "家戶綜所稅累稅率5%以下。", "家戶綜所稅累稅率百分之五以下。"),
    RegressionCase("news_cts_terms", "報稅讓岳父大人笑不出來。", "報稅讓丈人爸大人笑不出來。"),
    RegressionCase("news_cts_terms", "第三類農漁會", "第三類農漁會"),
    RegressionCase("news_cts_terms", "第六類區公所", "第六類區公所"),
    RegressionCase("news_cts_terms", "第一類就是投保子女公司行號", "第一類就是投保子女公司行號"),
    RegressionCase("news_cts_terms", "低收入戶中低收入戶", "低收入戶中低收入戶"),
    RegressionCase("news_cts_terms", "中低收入老人的生活津貼", "中低收入老人的生活津貼"),
    RegressionCase("news_cts_terms", "家庭應計人口總收入與資產", "家庭應計人口總收入佮資產"),
    RegressionCase("news_cts_terms", "拋棄繼承前", "拋棄繼承前"),
    RegressionCase("news_cts_terms", "未滿70歲直系親屬", "未滿七十歲直系親屬"),
    RegressionCase("news_cts_terms", "滿70歲直系親屬", "滿七十歲直系親屬"),
    RegressionCase("news_cts_terms", "最重可處1到4年刑期", "最重可處一到四年刑期"),
    RegressionCase("news_cts_terms", "2026台北市長選戰", "二千零二十六台北市長選戰"),
    RegressionCase("news_cts_terms", "台北市長選戰", "台北市長選戰"),
    RegressionCase("news_cts_terms", "操盤政策規劃", "操盤政策規劃"),
    RegressionCase("news_cts_terms", "北北基的總督導", "北北基的總督導"),
    RegressionCase("news_cts_terms", "各自的節奏", "各自的節奏"),
    RegressionCase("news_cts_terms", "縣市首長選戰", "縣市首長選戰"),
    RegressionCase("news_cts_terms", "總統大選前哨戰", "總統大選前哨戰"),
    RegressionCase("news_cts_terms", "飲食權益", "飲食權益"),
    RegressionCase("news_cts_terms", "學生飲食權益", "學生飲食權益"),
    RegressionCase("news_cts_terms", "攸關全縣學生飲食權益", "攸關全縣學生飲食權益"),
    RegressionCase("news_cts_terms", "履約能力資格限制", "履約能力資格限制"),
    RegressionCase("news_cts_terms", "調降投標門檻", "調降投標門檻"),
    RegressionCase("news_cts_terms", "夫妻檔業者", "夫妻檔業者"),
    RegressionCase("news_cts_terms", "貪污圖利罪", "貪污圖利罪"),
    RegressionCase("news_cts_terms", "司法偵辦階段", "司法偵辦階段"),
    RegressionCase("news_cts_terms", "蒐證行動", "蒐證行動"),
    RegressionCase("news_cts_terms", "巨額採購", "巨額採購"),
    RegressionCase("news_cts_terms", "2022年", "二千零二十二年"),
    RegressionCase("news_cts_terms", "14日", "十四日"),
    RegressionCase("news_cts_terms", "鄭麗文訪美行程", "鄭麗文訪美行程"),
    RegressionCase("news_cts_terms", "僑胞落地招待", "僑胞落地招待"),
    RegressionCase("news_cts_terms", "落地招待", "落地招待"),
    RegressionCase("news_cts_terms", "國民黨文傳會主委", "國民黨文傳會主委"),
    RegressionCase("news_cts_terms", "政黨補助款", "政黨補助款"),
    RegressionCase("news_cts_terms", "彰化縣長選舉", "彰化縣長選舉"),
    RegressionCase("news_cts_terms", "國民黨中常會", "國民黨中常會"),
    RegressionCase("news_cts_terms", "前考紀會主委", "前考紀會主委"),
    RegressionCase("news_cts_terms", "脫黨參選", "脫黨參選"),
    RegressionCase("news_cts_terms", "不回答假設性問題", "不回答假設性問題"),
    RegressionCase("news_cts_terms", "空降沒有行政經驗", "空降沒有行政經驗"),
    RegressionCase("news_cts_terms", "彰化過去從來沒有空降成功勝選過", "彰化過去從來沒有空降成功勝選過"),
    RegressionCase("news_cts_terms", "輝達執行長黃仁勳", "輝達執行長黃仁勳"),
    RegressionCase("news_cts_terms", "支持美國與川普政府施政目標", "支持美國佮川普政府施政目標"),
    RegressionCase("news_cts_terms", "空軍一號", "空軍一號"),
    RegressionCase("news_cts_terms", "北京天壇公園", "北京天壇公園"),
    RegressionCase("news_cts_terms", "北京四季酒店", "北京四季酒店"),
    RegressionCase("news_cts_terms", "13日到14日暫停開放", "十三日到十四日暫停開放"),
    RegressionCase("news_cts_terms", "預計14日下午參觀", "預計十四日下晝參觀"),
    RegressionCase("news_cts_terms", "從12到15日都不接受預訂", "對十二到十五日都不接受預訂"),
    RegressionCase("news_cts_terms", "救出我父親的人", "救出我父親的人"),
    RegressionCase("news_cts_terms", "航空追蹤網站", "航空追蹤網站"),
    RegressionCase("news_cts_terms", "末日飛機", "末日飛機"),
    RegressionCase("news_cts_terms", "美4月通膨攀升至3年來高峰", "美四月通膨攀升至三年來高峰"),
    RegressionCase("news_cts_terms", "通膨攀升至3年來高峰", "通膨攀升至三年來高峰"),
    RegressionCase("news_cts_terms", "三年來高峰", "三年來高峰"),
    RegressionCase("news_cts_terms", "范斯改口坦承不理想", "范斯改口坦承不理想"),
    RegressionCase("news_cts_terms", "不理想", "不理想"),
    RegressionCase("news_cts_terms", "阿聯否認尼坦雅胡密訪傳聞", "阿聯否認尼坦雅胡密訪傳聞"),
    RegressionCase("news_cts_terms", "雙邊關係公開透明", "雙邊關係公開透明"),
    RegressionCase("news_cts_terms", "巴西媒體關注美中經貿合作", "巴西媒體關注美中經貿合作"),
    RegressionCase("news_cts_terms", "陳思宏赴斯洛伐克演講", "陳思宏赴斯洛伐克演講"),
    RegressionCase("news_cts_terms", "深化中歐讀者理解台灣文化", "深化中歐讀者理解台灣文化"),
    RegressionCase("news_cts_terms", "台灣再度未獲WHA邀請", "台灣再度未獲WHA邀請"),
    RegressionCase("news_cts_terms", "美挺台有意義參與國際組織", "美挺台有意義參與國際組織"),
    RegressionCase("news_cts_terms", "有意義參與國際組織", "有意義參與國際組織"),
    RegressionCase("news_cts_terms", "AI樂觀情緒蓋過通膨利空", "AI樂觀情緒蓋過通膨利空"),
    RegressionCase("news_cts_terms", "樂觀情緒", "樂觀情緒"),
    RegressionCase("news_cts_terms", "標普那指收盤創新高", "標普那指收盤創新高"),
    RegressionCase("news_cts_terms", "收盤創新高", "收盤創新高"),
    RegressionCase("news_cts_terms", "國際油價震盪走低", "國際油價震盪走低"),
    RegressionCase("news_cts_terms", "台灣是模範盟友非交易籌碼", "台灣是模範盟友非交易籌碼"),
    RegressionCase("news_cts_terms", "台中榮總2醫師遭求重刑", "台中榮總二醫師遭求重刑"),
    RegressionCase("news_cts_terms", "雷達回波通過北台灣卻沒雨", "雷達回波通過北台灣卻沒雨"),
    RegressionCase("news_cts_terms", "雷達亮帶現象", "雷達亮帶現象"),
    RegressionCase("news_cts_terms", "巴黎發明展台灣奪11金", "巴黎發明展台灣奪十一金"),
    RegressionCase("news_cts_terms", "磁吸式卡踏獲獎", "磁吸式卡踏獲獎"),
    RegressionCase("news_cts_terms", "2026宜蘭童玩節7/4登場", "二千零二十六宜蘭童玩節七月四日登場"),
    RegressionCase("news_cts_terms", "宜蘭童玩節", "宜蘭童玩節"),
    RegressionCase("news_cts_terms", "7/4登場", "七月四日登場"),
    RegressionCase("news_cts_terms", "預售票5/1開賣享優惠", "預售票五月一日開賣享優惠"),
    RegressionCase("news_cts_terms", "5/1開賣", "五月一日開賣"),
    RegressionCase("news_cts_terms", "立院外交團倫敦會僑界", "立院外交團倫敦會僑界"),
    RegressionCase("news_cts_terms", "韓國瑜稱許日不落台商", "韓國瑜稱許日不落台商"),
    RegressionCase("news_cts_terms", "日不落台商", "日不落台商"),
    RegressionCase("news_cts_terms", "川習會在即盧比歐", "川習會在即盧比歐"),
    RegressionCase("news_cts_terms", "美國希望說服中國勸阻伊朗", "美國希望說服中國勸阻伊朗"),
    RegressionCase("news_cts_terms", "川習會在即台灣議題受矚", "川習會在即台灣議題受矚"),
    RegressionCase("news_cts_terms", "前美外交官籲堅守對台政策", "前美外交官籲堅守對台政策"),
    RegressionCase("news_cts_terms", "川普提名人選過關", "川普提名人選過關"),
    RegressionCase("news_cts_terms", "美參院確認華許接掌聯準會", "美參院確認華許接掌聯準會"),
    RegressionCase("news_cts_terms", "川習會前市場觀望", "川習會前市場觀望"),
    RegressionCase("news_cts_terms", "川習會前美退將撰文", "川習會前美退將撰文"),
    RegressionCase("news_cts_terms", "週刊爆籃籃蹭票看WBC", "週刊爆籃籃蹭票看WBC"),
    RegressionCase("news_cts_terms", "胡瓜淪掏錢工具人寒心", "胡瓜淪掏錢工具人寒心"),
    RegressionCase("news_cts_terms", "掏錢工具人", "掏錢工具人"),
    RegressionCase("news_cts_terms", "第37屆金曲獎", "第三十七屆金曲獎"),
    RegressionCase("news_cts_terms", "完整入圍名單一次看", "完整入圍名單一次看"),
    RegressionCase("news_cts_terms", "普發石油津貼1萬", "普發石油津貼一萬"),
    RegressionCase("news_cts_terms", "主計長油價凍漲優先", "主計長油價凍漲優先"),
    RegressionCase("news_cts_terms", "二千零二十六宜蘭童玩節七/四登場", "二千零二十六宜蘭童玩節七月四日登場"),
    RegressionCase("news_cts_terms", "七/四登場", "七月四日登場"),
    RegressionCase("news_cts_terms", "預售票五/一開賣享優惠", "預售票五月一日開賣享優惠"),
    RegressionCase("news_cts_terms", "五/一開賣", "五月一日開賣"),
    RegressionCase("news_cts_terms", "童沒帶備用衣被拍打後背淤傷", "童沒帶備用衣被拍打後背淤傷"),
    RegressionCase("news_cts_terms", "沒帶備用衣", "沒帶備用衣"),
    RegressionCase("news_cts_terms", "被拍打後背淤傷", "被拍打後背淤傷"),
    RegressionCase("news_cts_terms", "戶外活動的備用衣服", "戶外活動的備用衣服"),
    RegressionCase("news_cts_terms", "左後背處一大片紅腫", "左後背處一大片紅腫"),
    RegressionCase("news_cts_terms", "第一時間進行通報", "第一時間進行通報"),
    RegressionCase("news_cts_terms", "已經將這名老師解聘", "已經將這名老師解聘"),
    RegressionCase("news_cts_terms", "接獲通報後隨即展開調查", "接獲通報後隨即展開調查"),
    RegressionCase("news_cts_terms", "本案若查證屬實", "本案若查證屬實"),
    RegressionCase("news_cts_terms", "涉案人會面臨最高新台幣60萬的罰鍰", "涉案人會面臨上懸新台幣六十萬的罰鍰"),
    RegressionCase("news_cts_terms", "涉案人會面臨最高新台幣六十萬的罰鍰", "涉案人會面臨上懸新台幣六十萬的罰鍰"),
    RegressionCase("news_cts_terms", "針對涉案機構", "針對涉案機構"),
    RegressionCase("news_cts_terms", "視情節輕重", "視情節輕重"),
    RegressionCase("news_cts_terms", "減招停招停辦或廢止設立許可", "減招停招停辦或廢止設立許可"),
    RegressionCase("news_cts_terms", "女兒已經害怕上學", "女兒已經害怕上學"),
    RegressionCase("news_cts_terms", "心裡的陰影", "心裡的陰影"),
    RegressionCase("news_cts_terms", "短時間還無法抹去", "短時間還無法抹去"),
    RegressionCase("news_cts_terms", "登門道歉", "登門道歉"),
    RegressionCase("news_cts_terms", "承認自己失控認錯", "承認自己失控認錯"),
    RegressionCase("news_cts_terms", "耳朵後方也是瘀傷還脫皮", "耳朵後方也是瘀傷還脫皮"),
    RegressionCase("news_cts_terms", "遭指控幼兒園主任", "遭指控幼兒園主任"),
    RegressionCase("news_cts_terms", "民進黨今提名沈伯洋戰北市長對決蔣萬安", "民進黨今提名沈伯洋戰北市長對決蔣萬安"),
    RegressionCase("news_cts_terms", "沈伯洋戰北市長對決蔣萬安", "沈伯洋戰北市長對決蔣萬安"),
    RegressionCase("news_cts_terms", "中執會後將正式提名", "中執會後將正式提名"),
    RegressionCase("news_cts_terms", "立委沈伯洋出戰", "立委沈伯洋出戰"),
    RegressionCase("news_cts_terms", "操盤政策布局", "操盤政策布局"),
    RegressionCase("news_cts_terms", "透過數據整理", "透過數據整理"),
    RegressionCase("news_cts_terms", "新竹縣的提名進度", "新竹縣的提名進度"),
    RegressionCase("news_cts_terms", "竹北市長鄭朝方點頭", "竹北市長鄭朝方點頭"),
    RegressionCase("news_cts_terms", "目前普遍認為在等竹北市長鄭朝方點頭", "目前普遍認為在等竹北市長鄭朝方點頭"),
    RegressionCase("news_cts_terms", "民進黨選對會共同召集人", "民進黨選對會共同召集人"),
    RegressionCase("news_cts_terms", "快閃進入中央黨部", "快閃進入中央黨部"),
    RegressionCase("news_cts_terms", "正式披綠袍上陣", "正式披綠袍上陣"),
    RegressionCase("news_cts_terms", "對決現任市長蔣萬安", "對決現任市長蔣萬安"),
    RegressionCase("news_cts_terms", "柯文哲執政8年", "柯文哲執政八年"),
    RegressionCase("news_cts_terms", "蔣萬安2年多", "蔣萬安二年多"),
    RegressionCase("news_cts_terms", "台北隊成形", "台北隊成形"),
    RegressionCase("news_cts_terms", "合體看板", "合體看板"),
    RegressionCase("news_cts_terms", "民進黨全台布局本島就差新竹縣", "民進黨全台布局本島就差新竹縣"),
    RegressionCase("news_cts_terms", "禮讓無黨籍的花蓮縣議長張峻", "禮讓無黨籍的花蓮縣議長張峻"),
    RegressionCase("news_cts_terms", "離島金門連江目前也還沒拍板", "離島金門連江目前也還沒拍板"),
    RegressionCase("news_cts_terms", "早早提名的澎湖", "早早提名的澎湖"),
    RegressionCase("news_cts_terms", "有些地方慢一點", "有些地方慢一點"),
    RegressionCase("news_cts_terms", "找到適合的公僕為當地服務", "找到適合的公僕為當地服務"),
    RegressionCase("news_cts_terms", "力拚鞏固政治版圖", "力拚鞏固政治版圖"),
    RegressionCase("news_cts_terms", "2028總統大選前哨戰", "二千零二十八總統大選前哨戰"),
    RegressionCase("news_cts_terms", "花蓮免費營養午餐爆弊案", "花蓮免費營養午餐爆弊案"),
    RegressionCase("news_cts_terms", "議員前科長等5至30萬交保", "議員前科長等五至三十萬交保"),
    RegressionCase("news_cts_terms", "花蓮縣政府推動多年的免費營養午餐政策", "花蓮縣政府推動多年的免費營養午餐政策"),
    RegressionCase("news_cts_terms", "檢調昨(12)日搜索教育處", "檢調昨(十二)日搜索教育處"),
    RegressionCase("news_cts_terms", "約談營養師前科長花蓮縣議員吳東昇", "約談營養師前科長花蓮縣議員吳東昇"),
    RegressionCase("news_cts_terms", "訊後依貪污罪", "訊後依貪污罪"),
    RegressionCase("news_cts_terms", "諭知分別以五萬到三十萬元交保", "諭知分別以五萬到三十萬元交保"),
    RegressionCase("news_cts_terms", "檢方大動作調查", "檢方大動作調查"),
    RegressionCase("news_cts_terms", "四年前的招標案", "四年前的招標案"),
    RegressionCase("news_cts_terms", "涉嫌圖利特定廠商", "涉嫌圖利特定廠商"),
    RegressionCase("news_cts_terms", "不排除向上追查擴大偵辦", "不排除向上追查擴大偵辦"),
    RegressionCase("news_cts_terms", "政風回報是行政疏失", "政風回報是行政疏失"),
    RegressionCase("news_cts_terms", "全力配合檢調來釐清事實", "全力配合檢調來釐清事實"),
    RegressionCase("news_cts_terms", "全縣超過兩萬名學生的飲食", "全縣超過兩萬名學生的飲食"),
    RegressionCase("news_cts_terms", "年度經費高達2億7千萬元", "年度經費高達二億七千萬元"),
    RegressionCase("news_cts_terms", "攸關全縣2萬1189名學生飲食權益", "攸關全縣二萬一千一百八十九名學生飲食權益"),
    RegressionCase("news_cts_terms", "七家廠商投標", "七家廠商投標"),
    RegressionCase("news_cts_terms", "前一年度得標業者", "前一年度得標業者"),
    RegressionCase("news_cts_terms", "第七家廠商因為資格不符", "第七家廠商因為資格不符"),
    RegressionCase("news_cts_terms", "依法應該取消投標資格", "依法應該取消投標資格"),
    RegressionCase("news_cts_terms", "教育處後續卻撤銷原本的兩大標案", "教育處後續卻撤銷原本的兩大標案"),
    RegressionCase("news_cts_terms", "重新拆分成十個小標案", "重新拆分成十個小標案"),
    RegressionCase("news_cts_terms", "民意代表介入", "民意代表介入"),
    RegressionCase("news_cts_terms", "整個過程都是依法辦理", "整個過程都是依法辦理"),
    RegressionCase("news_cts_terms", "維護學生的食安權益", "維護學生的食安權益"),
    RegressionCase("news_cts_terms", "終身不得再進入教保機構任職。", "終身不得再進入教保機構任職。"),
    RegressionCase("news_cts_terms", "最高新台幣60萬元罰鍰。", "上懸新台幣六十萬元罰鍰。"),
    RegressionCase("news_cts_terms", "減招停招停辦。", "減招停招停辦。"),
    RegressionCase("news_cts_terms", "廢止設立許可。", "廢止設立許可。"),
    RegressionCase("news_cts_terms", "老人健保補助9912元。", "老人健保補助九千九百一十二元。"),
    RegressionCase("news_cts_terms", "健保補助入戶頭。", "健保補助入戶頭。"),
    RegressionCase("news_cts_terms", "社會福利補助。", "社會福利補助。"),
    RegressionCase("news_cts_terms", "身心障礙者生活補助。", "身心障礙者生活補助。"),
    RegressionCase("news_cts_terms", "特定長照補助。", "特定長照補助。"),
    RegressionCase("news_cts_terms", "補助條件喪失。", "補助條件喪失。"),
    RegressionCase("news_cts_terms", "影響所及只有四個縣市。", "影響所及只有四个縣市。"),
    # scheduling / plans
    RegressionCase("schedule_plans", "等一下打給你。", "等陣仔閣敲予你。"),
    RegressionCase("schedule_plans", "改天再說。", "改工閣講。"),
    RegressionCase("schedule_plans", "早點休息。", "較早歇睏。"),
    RegressionCase("schedule_plans", "要轉車嗎？", "愛轉車無？"),
    RegressionCase("schedule_plans", "我等你在門口。", "我等你佇門跤口。"),
    RegressionCase("schedule_plans", "我現在在等車。", "我這馬咧等車。"),
    RegressionCase("schedule_plans", "你打算去哪邊游泳呢？", "你欲去佗位泅水咧？"),
    RegressionCase("schedule_plans", "你打算去哪座山呢？", "你欲去佗一座山咧？"),
    RegressionCase("schedule_plans", "我等了一個多月。", "我等一个外月。"),
    RegressionCase("schedule_plans", "路上小心。", "路裡細膩。"),
    RegressionCase("schedule_plans", "我晚點回你。", "我較晏閣回你。"),
    RegressionCase("schedule_plans", "我到了再打給你。", "我到矣閣敲予你。"),
    RegressionCase("schedule_plans", "晚點再聊。", "較晏閣聊。"),
    RegressionCase("schedule_plans", "我晚點再回覆你。", "我較晏閣回覆你。"),
    RegressionCase("schedule_plans", "我晚點再跟你說。", "我較晏閣共你講。"),
    RegressionCase("schedule_plans", "我晚點到你家。", "我較晏到你兜。"),
    RegressionCase("schedule_plans", "我等等再出門。", "我等陣仔閣出門。"),
    RegressionCase("schedule_plans", "我晚點再打給你。", "我較晏閣敲予你。"),
    RegressionCase("schedule_plans", "可以幫我取消預約嗎？", "會當替我取消預約無？"),
    RegressionCase("schedule_plans", "我想取消預約。", "我想欲取消預約。"),
    RegressionCase("schedule_plans", "我需要取消預約。", "我欲取消預約。"),
    RegressionCase("schedule_plans", "我想改接送時間。", "我想欲改接送時間。"),
    RegressionCase("schedule_plans", "我需要安排接送時間。", "我欲安排接送時間。"),
    RegressionCase("schedule_plans", "我可以幫你們安排接送時間。", "我會當替恁安排接送時間。"),
    RegressionCase("schedule_plans", "我可以幫他們安排接送時間。", "我會當替怹安排接送時間。"),
    RegressionCase("schedule_plans", "我可以幫大家安排接送時間。", "我會當替逐家安排接送時間。"),
    RegressionCase("schedule_plans", "我可以幫各位安排接送時間。", "我會當替逐家安排接送時間。"),
    RegressionCase("schedule_plans", "我可以幫客人安排接送時間。", "我會當替人客安排接送時間。"),
    RegressionCase("schedule_plans", "我可以幫病人安排接送時間。", "我會當替病人安排接送時間。"),
    RegressionCase("schedule_plans", "我可以幫家屬安排接送時間。", "我會當替家屬安排接送時間。"),
    RegressionCase("schedule_plans", "可以協助你們安排接送時間。", "會當鬥相共恁安排接送時間。"),
    RegressionCase("schedule_plans", "可以協助他們安排接送時間。", "會當鬥相共怹安排接送時間。"),
    RegressionCase("schedule_plans", "可以協助大家安排接送時間。", "會當鬥相共逐家安排接送時間。"),
    RegressionCase("schedule_plans", "可以協助各位安排接送時間。", "會當鬥相共逐家安排接送時間。"),
    RegressionCase("schedule_plans", "可以協助客人安排接送時間。", "會當鬥相共人客安排接送時間。"),
    RegressionCase("schedule_plans", "可以協助病人安排接送時間。", "會當鬥相共病人安排接送時間。"),
    RegressionCase("schedule_plans", "可以協助家屬安排接送時間。", "會當鬥相共家屬安排接送時間。"),
    RegressionCase("schedule_plans", "請協助安排接送時間。", "請鬥相共安排接送時間。"),
    RegressionCase("schedule_plans", "請協助我們安排接送時間。", "請鬥相共阮安排接送時間。"),
    RegressionCase("schedule_plans", "請幫我們安排接送時間。", "請替阮安排接送時間。"),
    RegressionCase("schedule_plans", "能不能協助我安排接送時間？", "敢會當鬥相共我安排接送時間？"),
    RegressionCase("schedule_plans", "能不能請您協助我安排接送時間？", "敢會當請你鬥相共我安排接送時間？"),
    RegressionCase("schedule_plans", "能不能請您協助我們安排接送時間？", "敢會當請你鬥相共阮安排接送時間？"),
    RegressionCase("schedule_plans", "可不可以協助我安排接送時間？", "敢會當鬥相共我安排接送時間？"),
    RegressionCase("schedule_plans", "麻煩幫我安排接送時間。", "麻煩替我安排接送時間。"),
    RegressionCase("schedule_plans", "幫我安排接送時間。", "替我安排接送時間。"),
    RegressionCase("schedule_plans", "可不可以幫我安排接送時間？", "敢會當替我安排接送時間？"),
    RegressionCase("schedule_plans", "可不可以幫我們安排接送時間？", "敢會當替阮安排接送時間？"),
    RegressionCase("schedule_plans", "可不可以幫你們安排接送時間？", "敢會當替恁安排接送時間？"),
    RegressionCase("schedule_plans", "可不可以幫他們安排接送時間？", "敢會當替怹安排接送時間？"),
    RegressionCase("schedule_plans", "可不可以幫大家安排接送時間？", "敢會當替逐家安排接送時間？"),
    RegressionCase("schedule_plans", "可不可以幫各位安排接送時間？", "敢會當替逐家安排接送時間？"),
    RegressionCase("schedule_plans", "可不可以幫客人安排接送時間？", "敢會當替人客安排接送時間？"),
    RegressionCase("schedule_plans", "可不可以幫病人安排接送時間？", "敢會當替病人安排接送時間？"),
    RegressionCase("schedule_plans", "可不可以幫家屬安排接送時間？", "敢會當替家屬安排接送時間？"),
    RegressionCase("schedule_plans", "能幫我安排接送時間嗎？", "會當替我安排接送時間無？"),
    RegressionCase("schedule_plans", "是否能幫我安排接送時間？", "敢會當替我安排接送時間？"),
    RegressionCase("schedule_plans", "方不方便幫我安排接送時間？", "敢方便替我安排接送時間？"),
    RegressionCase("schedule_plans", "方不方便協助我安排接送時間？", "敢方便鬥相共我安排接送時間？"),
    RegressionCase("schedule_plans", "方不方便協助我們安排接送時間？", "敢方便鬥相共阮安排接送時間？"),
    RegressionCase("schedule_plans", "方不方便請您幫我們安排接送時間？", "敢方便請你替阮安排接送時間？"),
    RegressionCase("schedule_plans", "可否麻煩你幫我安排接送時間？", "敢會當麻煩你替我安排接送時間？"),
    RegressionCase("schedule_plans", "拜託幫我安排接送時間。", "拜託替我安排接送時間。"),
    RegressionCase("schedule_plans", "希望你幫我安排接送時間。", "希望你替我安排接送時間。"),
    RegressionCase("schedule_plans", "是否方便請你幫我安排接送時間？", "敢方便請你替我安排接送時間？"),
    RegressionCase("schedule_plans", "可不可以麻煩你幫我安排接送時間？", "敢會當麻煩你替我安排接送時間？"),
    RegressionCase("schedule_plans", "可不可以麻煩您協助我安排接送時間？", "敢會當麻煩你鬥相共我安排接送時間？"),
    RegressionCase("schedule_plans", "可不可以麻煩您協助我們安排接送時間？", "敢會當麻煩你鬥相共阮安排接送時間？"),
    RegressionCase("schedule_plans", "可不可以麻煩您幫我們安排接送時間？", "敢會當麻煩你替阮安排接送時間？"),
    RegressionCase("schedule_plans", "我想改提醒時間。", "我想欲改提醒時間。"),
    RegressionCase("schedule_plans", "我想改集合時間。", "我想欲改集合時間。"),
    RegressionCase("schedule_plans", "我想改報到時間。", "我想欲改報到時間。"),
    RegressionCase("schedule_plans", "我想改聊天時間。", "我想欲改開講時間。"),
    RegressionCase("schedule_plans", "我想改見面時間。", "我想欲改見面時間。"),
    RegressionCase("schedule_plans", "我想改通話時間。", "我想欲改通話時間。"),
    RegressionCase("schedule_plans", "我想改聯絡時間。", "我想欲改聯絡時間。"),
    RegressionCase("schedule_plans", "我想改接送日期。", "我想欲改接送日期。"),
    RegressionCase("schedule_plans", "我想改聚會時間。", "我想欲改聚會時間。"),
]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="日常會話情境 regression runner")
    parser.add_argument("--rounds", type=int, default=1, help="重複跑幾輪 exact-match 回歸")
    parser.add_argument("--category", action="append", default=[], help="只跑指定 category，可重複傳入")
    parser.add_argument("--list-categories", action="store_true", help="列出所有 category")
    parser.add_argument("--show-pass", action="store_true", help="顯示每筆通過案例")
    parser.add_argument("--fail-fast", action="store_true", help="遇到第一個 mismatch 就停止")
    return parser


def _selected_cases(categories: list[str]) -> list[RegressionCase]:
    if not categories:
        return CONVERSATION_REGRESSION_CASES
    wanted = set(categories)
    return [case for case in CONVERSATION_REGRESSION_CASES if case.category in wanted]


def main() -> int:
    args = _build_parser().parse_args()
    categories = sorted({case.category for case in CONVERSATION_REGRESSION_CASES})
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
