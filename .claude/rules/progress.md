# 現況與下一步方向

> 精簡接手版。只保留目前狀態、最近重要變更與下一步；舊的逐輪細節已封存到 `.claude/rules/archive/progress-2026-05.md`。
> 除非需要追溯某輪詞條/規則的歷史原因，不要預設讀 archive，避免消耗大量 LLM token。

## 目前辭典狀態（2026-05-29）

| 類型 | 數量 |
|------|------|
| 總詞條（active runtime） | 20,458 |
| 人工驗證（trust: human active rows） | 12,374 |
| base seed（trust: seed，低信任） | source 仍有短詞 active；runtime 以 policy 過濾高風險 seed |
| 最新資料 round | round443 |

所有迴歸測試全部通過（共 3431 筆）：
bus 549、medical 220、transport 269、conversation 1163、restaurant 98、shopping 188、hotel 171、taxi 83、bank 146、school 110、family 88、workplace 346。

## 最近重要變更

### 2026-05-29：CTS 文夏經典金曲活動完整句正式語境補強（round443）

- 來源：華視台語新聞列表「總爺『聞到夏天』 文夏用音樂串聯新舊世代」（2026-05-28；2026-05-29 查驗）。
- 補強 `為紀念「寶島歌王」文夏老師一生對台灣的貢獻`、`透過兩天的活動，以不同主題的演出，串聯文夏、台南、麻豆、在地居民與青年創作者`、`有市集，有演唱會，還有我們的特展，多元的形式，來帶領大家進入文夏老師的音樂創作` 等 6 筆。
- 修正活動/文化新聞正式語境中 `一生→一世人`、`兩天→兩工`、`不同→無仝`、`還有我們的→閣有阮的`、`大家→逐家`、`更加→閣較` 等局部不合語境的輸出；`金曲歌後/歌后` 依既有字形正規化以長詞條穩定。
- `scripts/run_conversation_regression.py` 新增 6 筆 `news_cts_terms` case；12 支 regression 全綠。

### 2026-05-29：CTS 關西石油津貼設籍期限新聞正式語境補強（round442）

- 來源：華視台語新聞列表「全國首創『石油津貼』普發5千 關西湧遷戶籍人潮」（2026-05-27；2026-05-29 查驗，內文資格期限為 2026-05-29）。
- 補強 `全台首例「石油津貼」，出現在新竹縣關西鎮，每位鎮民普發5千元`、`只要在5月29日前設籍關西鎮，都符合領取資格`、`根據統計，25日、26日兩天，超過5百人遷戶籍到關西，也讓工作人員得加班到晚上8點` 等 5 筆。
- 修正正式/行政新聞語境中 `出現在→出現佇`、`每位→逐位`、`根據→照`、`兩天→兩工`、`晚上→暗時` 等不合語境的輸出；數字仍依既有管線正規化為漢字。
- `scripts/run_conversation_regression.py` 新增 5 筆 `news_cts_terms` case；12 支 regression 全綠。

### 2026-05-29：CTS 桃園迪士尼公車新聞情境補強（round441）

- 來源：華視台語新聞列表「下車鈴有『米奇』說掰掰！ 乘客讚主題公車撫慰人心」（2026-05-28；2026-05-29 查驗）。
- 補強 `但您聽過卡通角色跟您說掰掰嗎？`、`乘客不是趕著上車找座位，而是拿著手機在車內拍拍拍`、`知名IP融入通勤日常，讓公車不只是交通工具，也替城市增添童趣氛圍` 等 5 筆。
- 修正 `您→你`、`跟→佮`、`嗎→無`、`不只→毋但`、`車上→車頂`、`很多→誠濟`、`打氣→風筒`、`工具→家私` 等正式/新聞語境不合輸出。
- `scripts/run_conversation_regression.py` 新增 5 筆 `news_cts_terms` case；12 支 regression 全綠。

### 2026-05-29：CTS AIT 建國酒會外交新聞正式語境補強（round440）

- 來源：華視台語新聞列表「致詞暗喻台灣朝野？ 谷立言舉前總統故事省思『化解政敵』」（2026-05-28；2026-05-29 查驗）。
- 補強 `向來賓舉杯敬酒，總統賴清德，出席美國建國250週年酒會`、`藍綠兩黨主席王不見王，谷立言似乎也看在眼裡`、`台積電就是台灣的驕傲，就是台灣的經貿實力，可以讓美方的朋友一起認識台灣` 等 8 筆。
- 修正外交/正式新聞語境中 `向來賓→由來賓`、`還有→閣有`、`搜尋→搜揣`、`似乎→敢若`、`看在眼裡→看佇眼內底`、`他們→怹`、`書→冊`、`認識→熟似` 等不合語境的輸出。
- `scripts/run_conversation_regression.py` 新增 8 筆 `news_cts_terms` case；12 支 regression 全綠。

### 2026-05-29：CTS 寮國洞穴救援新聞國名邊界補強（round439）

- 來源：華視台語新聞列表「寮國尋金7人困洞一週 5人獲救激動落淚」（2026-05-28；2026-05-29 查驗）。
- 補強 `寮國一處洞穴傳出驚險救援`、`搜救人員24日展開行動後，終於找到其中5人，他們平安獲救後激動相擁`、`他們從24日開始，就展開搜救行動` 等 6 筆。
- 修正 `寮國一處` 被短詞 `國一→初一` 切成 `寮初一處` 的國名邊界錯誤；同時穩定救援流程句中的日期、人稱、`從`、`找到`、`成為` 等正式新聞語境。
- `scripts/run_conversation_regression.py` 新增 6 筆 `news_cts_terms` case；12 支 regression 全綠。

### 2026-05-29：CTS 漫畫店消防設備誤拆新聞正式語境補強（round438）

- 來源：華視台語新聞列表「偵煙警報器.針孔傻傻分不清？ 漫畫店設備遭強拆除」（2026-05-28；2026-05-29 查驗）。
- 補強 `近期醫美針孔偷拍案件人心惶惶，現在有民眾到高雄新興區中正二路一間漫畫店消費`、`消防設備竟被硬生生扯下，不只外殼塑膠斷裂，裡面的接線也被拆開`、`切勿自行強拆，除了會造成店家困擾、財物損失外，行為也已經觸法` 等 8 筆。
- 修正正式新聞/法律語境中 `現在→這馬`、`不只→毋但`、`裡面→內底`、`顧客→人客`、`以為→掠準`、`提高警覺→提懸警覺`、`行為→行踏` 等局部不合語境的輸出。
- `scripts/run_conversation_regression.py` 新增 8 筆 `news_cts_terms` case；12 支 regression 全綠。

### 2026-05-29：CTS 竹北市長初選新聞正式語境補強（round437）

- 來源：華視台語新聞列表「藍竹縣議員林禹佑參選竹北市長 鄭朝方：續拚市政」（2026-05-28；2026-05-29 查驗）。
- 補強 `帶您關注2026地方大選`、`國民黨新竹縣議員林禹佑，今(28)日上午在新竹縣黨部正式宣布參選竹北市長`、`國民黨還有新竹縣議員邱靖雅和吳旭智，以及竹北市民代表會主席林啟賢，都有意角逐` 等 6 筆。
- 修正正式新聞語境中 `您→你`、`上午→早起`、`宣布→宣佈`、`還有→閣有`、`誰→啥人`、`重要→要緊` 等局部不合語境的輸出。
- `scripts/run_conversation_regression.py` 新增 6 筆 `news_cts_terms` case；12 支 regression 全綠。

### 2026-05-29：CTS 台語新聞列表後段標題與引文補強（round436）

- 來源：華視台語新聞列表（2026-05-28 更新；2026-05-29 查驗）。
- 補強 `傅拋「生1胎免費住社宅30年.2胎終身」 綠轟激進`、`「微型電動車」高鐵站充電 違法！警通知車主：開罰`、`蔡壁如錄影談黨機制 柯文哲留言轟「來這放話？」` 等列表後段標題變體。
- 修正 `30年.2胎` 中半形句點後數字未正規化的標題輸出，並補 `中央委員不在中央黨部講，來這裡放話嗎`、`事情是發生在27日下午6點多` 的新聞語境穩定輸出。
- `scripts/run_conversation_regression.py` 新增 6 筆 `news_cts_terms` case；12 支 regression 全綠。

### 2026-05-29：CTS 台語新聞標題標點變體補強（round435）

- 來源：華視台語新聞列表（2026-05-28 更新；2026-05-29 查驗）。
- 補強直接貼新聞列表標題時的實際版面變體，包含 `總爺「聞到夏天」 文夏用音樂串聯新舊世代`、`台中「3大美食地圖」曝 APP帶你從早吃到晚`、`下車鈴有「米奇」說掰掰！ 乘客讚主題公車撫慰人心` 等 7 筆。
- 修正長詞條因引號、問號、驚嘆號、半形句點、空白與數字正規化後才產生的形態而不命中的情況。
- `scripts/run_conversation_regression.py` 新增 7 筆 `news_cts_terms` case；12 支 regression 全綠。

### 2026-05-28：CTS 台語新聞詞補強（round425-round434）

- 來源：華視台語新聞列表，多篇 2026-05-28 台語新聞。
- 補強 AIT 酒會、寮國洞穴救援、桃園迪士尼公車、竹北市長初選、漫畫店消防設備誤拆、文夏活動、美食地圖等新聞詞與長片語。
- 代表詞條包含 `寮國尋金7人困洞一週`、`偵煙警報器與偵熱感應器`、`台中3大美食地圖曝APP帶你從早吃到晚`、`昨27日晚上AIT舉行美國建國250週年酒會`、`讓公車不只是交通工具也替城市增添童趣氛圍`。
- 詳細逐條內容留在 `data/lexicon_entries.jsonl` 與 regression cases。

### 2026-05-28：CTS 文夏經典金曲活動細節詞補強（round424）

- 來源：華視台語新聞「總爺『聞到夏天』 文夏用音樂串聯新舊世代」（2026-05-28）。
- 補強 `以老中青三代的音樂人一起作為主題`、`將文夏的音樂作品重新、和地方串聯`、`週日則以年輕人演唱A Cappella的方式` 等同篇新聞細節片語。
- 修正 `一起作為主題` 被 core `一起→鬥陣` 轉成較突兀的 `鬥陣作為主題`，改為 `作伙作為主題`；也補上 `和地方串聯→佮地方串聯`。
- `scripts/run_conversation_regression.py` 新增 5 筆 `news_cts_terms` case；12 支 regression 全綠。

### 2026-05-23：CTS 無菸城市與南港城市治理詞補強（round410）

- 來源：華視台語新聞「無菸城藍綠轟！」，內容涵蓋台北市無菸城市政策、議會質詢、南港城市治理與首都市長戰。
- 補強 `無菸城藍綠轟`、`無菸城市政策`、`社群平台`、`辦公孤島`、`黑鄉變成潮城`、`首都市長戰的攻防之一` 等正式新聞詞。
- 修正正式語境被口語化或誤轉的問題，例如 `他的→伊的`、`先前→原前`、`高價→懸價`、`晚上→暗時`、`從→對`、`攻防→柔道`。
- `scripts/run_conversation_regression.py` 新增 19 筆 `news_cts_terms` case；12 支 regression 全綠。

### 2026-05-23：CTS 安平廟埕震天祭詞補強（round409）

- 來源：華視台語新聞「安平廟埕震天祭」明登場，解謎走讀認識信仰。
- 補強 `安平廟埕震天祭`、`安平開台天后宮`、`角頭廟宇`、`實境解謎走讀`、`城隍爺夫人`、`踩街嘉年華` 等活動/信仰詞。
- 修正新聞正式語境被口語化的問題，例如 `這次→這擺`、`我們→咱`、`跟→佮`、`其實→論真`、`這個→這个`、`將在→欲在`。
- `scripts/run_conversation_regression.py` 新增 18 筆 `news_cts_terms` case；12 支 regression 全綠。

### 2026-05-23：converter runtime 效能優化

- `_mask_protected_terms()` 減少重複 runtime phrase overlap 檢查。
- `_apply_rules()` 載入時排除 disabled/空 pattern 規則；literal 和純文字 regex 先用 `in` 快速判斷。
- 300 次混合句 benchmark：平均約 `0.330 ms` 降到 `0.285 ms`，p95 約 `0.478 ms` 降到 `0.421 ms`。
- 12 支 regression 全綠。

### 2026-05-23：README 補資料檔案使用順序

- README 新增 `資料檔案使用順序與功能`。
- 說明 `lexicon_entries.jsonl`、`rule_entries.jsonl`、`core_lexicon.json`、`char_verified_allowlist.txt`、`data/artifacts/*` 的使用順序與功能。

### 2026-05-23：CTS 新聞詞補強 round404-round410

- 補強華視台語新聞相關專有詞、地名、人名、機關名與新聞語境 passthrough。
- 最新 round410：無菸城市與南港城市治理詞補強。
- round409：安平廟埕震天祭與解謎走讀活動詞補強。
- round408：花蓮免費營養午餐採購弊案詞補強。
- round407：彰化縣長提名與藍營選戰詞補強。
- round406：馬英九基金會人事風波與調查聲明詞補強。
- round405：北市公車自撞山壁事故詞補強。
- round404：前鎮百貨系財神殿與參拜詞補強。
- 詳細逐條新增詞條、誤轉修正與來源 URL 請看 archive。

## 維護原則

1. 優先改 `data/lexicon_entries.jsonl`：單詞、片語、句子轉錯時先補詞條。
2. 謹慎改 `data/rule_entries.jsonl`：只有多句共享同一個穩定文法模式時才動規則。
3. 不手動改 `data/artifacts/*`：改來源資料後重編 artifacts。
4. 修改根目錄 `data/` 後，同步 `taigi_converter/data/` 並重編兩邊 artifacts。
5. `progress.md` 只保留接手摘要；大段歷史移到 `.claude/rules/archive/`。

## 常用驗證

```bash
python3 -m py_compile artifact_compiler.py converter.py app.py scripts/build_runtime_artifacts.py
python3 scripts/run_bus_regression.py
python3 scripts/run_medical_regression.py
python3 scripts/run_transport_regression.py
python3 scripts/run_conversation_regression.py
python3 scripts/run_restaurant_regression.py
python3 scripts/run_shopping_regression.py
python3 scripts/run_hotel_regression.py
python3 scripts/run_taxi_regression.py
python3 scripts/run_bank_regression.py
python3 scripts/run_school_regression.py
python3 scripts/run_family_regression.py
python3 scripts/run_workplace_regression.py
```

## 下一步方向

- 若持續做新聞詞補強，維持「來源 URL + 實測誤轉 + regression case」格式，但詳細內容放 archive。
- 若 runtime 資料量繼續變大，優先觀察 cold start、protected masking、regex rule 數量。
- 若 `progress.md` 超過約 150 行，再次整理舊內容進 archive。
