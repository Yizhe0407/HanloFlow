# 現況與下一步方向

> 每次大規模優化後請更新此檔案。這是讓下一個人快速接手的地圖。

## 目前辭典狀態（2026-05-08）

| 類型 | 數量 |
|------|------|
| 總詞條（active runtime） | 16,611 |
| 人工驗證（trust: human active rows） | ~7,638 |
| base seed（trust: seed，低信任） | source 仍有短詞 active；runtime 以 policy 過濾高風險 seed |
| 最新 round | round220 |

所有迴歸測試全部通過（共 1102 筆）：
bus 287、medical 99、transport 80、conversation 55、restaurant 72、shopping 88、hotel 83、taxi 68、bank 67、school 69、family 66、workplace 68

---

## 近期做了什麼（最新在前）

### 2026-05-08：飯店/醫療/購物/交通與計程車服務句補強（round220）
**新增人工詞條（19 條 active）：**
- `可以幫我查停車費嗎→會當替我查停車費無`
- `可以幫我改入住人數嗎→會當替我改入住人數無`
- `我想改加床數量→我想欲改加床數量`
- `可以幫我查洗衣時間嗎→會當替我查洗衫時間無`
- `可以幫我查掛號費嗎→會當替我查掛號費無`
- `我想改抽血時間→我想欲改抽血時間`
- `可以幫我查疫苗紀錄嗎→會當替我查疫苗紀錄無`
- `可以幫我改陪病人數嗎→會當替我改陪病人數無`
- `可以幫我查會員點數嗎→會當替我查會員點數無`
- `我想改取貨時間→我想欲改取貨時間`
- `可以幫我查退貨進度嗎→會當替我查退貨進度無`
- `可以幫我改商品顏色嗎→會當替我改商品顏色無`
- `可以幫我查票券效期嗎→會當替我查票券效期無`
- `我想改搭客運→我想欲改搭客運`
- `可以幫我查出口位置嗎→會當替我查出口位置無`
- `可以幫我改下車站嗎→會當替我改落車站無`
- `可以幫我查車子位置嗎→會當替我查車子位置無`
- `可以幫我改下車地址嗎→會當替我改落車地址無`
- `可以幫我查預估車資嗎→會當替我查預估車錢無`

**修正實測錯誤：**
- 飯店、醫療、購物、交通與計程車查詢請託 `可以幫我...` 收斂為 `會當替我...`
- `我想改加床數量。`、`我想改抽血時間。`、`我想改取貨時間。`、`我想改搭客運。` 補足 `想欲`
- 飯店洗衣語境固定 `洗衣時間→洗衫時間`
- 交通與計程車下車語境固定為 `落車`，車資固定為 `車錢`

**新增迴歸測試：**
- `run_hotel_regression.py / reservation` +2，分類合計 18 筆
- `run_hotel_regression.py / check_in` +1，分類合計 12 筆
- `run_hotel_regression.py / amenities` +1，分類合計 24 筆
- `run_medical_regression.py / registration` +1，分類合計 22 筆
- `run_medical_regression.py / tests` +1，分類合計 13 筆
- `run_medical_regression.py / pharmacy_payment` +1，分類合計 15 筆
- `run_medical_regression.py / rooms_inpatient` +1，分類合計 14 筆
- `run_shopping_regression.py / purchase` +1，分類合計 16 筆
- `run_shopping_regression.py / payment` +1，分類合計 14 筆
- `run_shopping_regression.py / after_sales` +2，分類合計 32 筆
- `run_transport_regression.py / platform_nav` +1，分類合計 11 筆
- `run_transport_regression.py / ticketing` +2，分類合計 19 筆
- `run_transport_regression.py / service_redirect` +1，分類合計 13 筆
- `run_taxi_regression.py / destination` +1，分類合計 17 筆
- `run_taxi_regression.py / payment` +1，分類合計 10 筆
- `run_taxi_regression.py / misc` +1，分類合計 16 筆
- hotel 總數 79→83
- medical 總數 95→99
- shopping 總數 84→88
- transport 總數 76→80
- taxi 總數 65→68

**資料同步：**
- 根目錄 `data/` 與 `taigi_converter/data/` 同步更新
- 重編根目錄與 package 內 artifacts

### 2026-05-08：餐廳/家庭/學校/辦公/銀行與公車服務句補強（round219）
**新增人工詞條（19 條 active）：**
- `可以幫我查候位進度嗎→會當替我查候位進度無`
- `可以幫我改用內用嗎→會當替我改做內用無`
- `我想改用現金付款→我想欲改用現錢付款`
- `可以幫我查外送進度嗎→會當替我查外送進度無`
- `可以幫我查小孩體溫嗎→會當替我查囡仔體溫無`
- `我想改晚睡時間→我想欲改晚睡時間`
- `可以幫我提醒爸爸嗎→會當替我提醒阿爸無`
- `我想改接小孩時間→我想欲改接囡仔時間`
- `可以幫我查補課時間嗎→會當替我查補課時間無`
- `我想改補考日期→我想欲改補考日期`
- `可以幫我通知同學嗎→會當替我通知同學無`
- `我想改作業期限→我想欲改作業期限`
- `可以幫我查簽核進度嗎→會當替我查簽核進度無`
- `我想改會議時間→我想欲改會議時間`
- `可以幫我查帳戶狀態嗎→會當替我查口座狀態無`
- `我想改領錢金額→我想欲改領錢金額`
- `可以幫我查公車路線嗎→會當替我查公車路線無`
- `我想改轉車地點→我想欲改轉車地點`
- `可以幫我查站牌位置嗎→會當替我查站牌位置無`

**修正實測錯誤：**
- 餐廳、家庭、學校、辦公、銀行與公車查詢請託 `可以幫我...` 收斂為 `會當替我...`
- `我想改用現金付款。`、`我想改接小孩時間。`、`我想改補考日期。`、`我想改作業期限。`、`我想改會議時間。`、`我想改領錢金額。`、`我想改轉車地點。` 補足 `想欲`
- 家庭親子語境固定 `小孩→囡仔`、`爸爸→阿爸`
- 銀行帳戶語境固定為 `口座`，避免 `帳戶狀態` 被拆成低品質輸出
- 餐廳內用切換固定為 `改做內用`

**新增迴歸測試：**
- `run_restaurant_regression.py / ordering` +1，分類合計 20 筆
- `run_restaurant_regression.py / seating` +1，分類合計 13 筆
- `run_restaurant_regression.py / payment` +1，分類合計 11 筆
- `run_restaurant_regression.py / service` +1，分類合計 18 筆
- `run_family_regression.py / parent_child` +1，分類合計 13 筆
- `run_family_regression.py / health_care` +1，分類合計 15 筆
- `run_family_regression.py / daily` +2，分類合計 21 筆
- `run_school_regression.py / teacher_class` +1，分類合計 14 筆
- `run_school_regression.py / student_class` +1，分類合計 19 筆
- `run_school_regression.py / homework` +1，分類合計 11 筆
- `run_school_regression.py / exam` +1，分類合計 11 筆
- `run_workplace_regression.py / meeting` +1，分類合計 20 筆
- `run_workplace_regression.py / workflow` +1，分類合計 18 筆
- `run_bank_regression.py / bank_account` +1，分類合計 19 筆
- `run_bank_regression.py / bank_transaction` +1，分類合計 21 筆
- `run_bus_regression.py / service_redirect` +1，分類合計 13 筆
- `run_bus_regression.py / route_transfer` +1，分類合計 16 筆
- `run_bus_regression.py / misc` +1，分類合計 8 筆
- restaurant 總數 68→72
- family 總數 62→66
- school 總數 65→69
- workplace 總數 66→68
- bank 總數 65→67
- bus 總數 284→287

**資料同步：**
- 根目錄 `data/` 與 `taigi_converter/data/` 同步更新
- 重編根目錄與 package 內 artifacts

### 2026-05-08：飯店/醫療/購物/交通與計程車服務句補強（round218）
**新增人工詞條（15 條 active）：**
- `可以幫我查早餐時間嗎→會當替我查早頓時間無`
- `可以幫我改住宿天數嗎→會當替我改歇暝天數無`
- `我想改房型→我想欲改房型`
- `可以幫我查房間設備嗎→會當替我查房間設備無`
- `可以幫我查藥局位置嗎→會當替我查藥局位置無`
- `我想改看診科別→我想欲改看診科別`
- `可以幫我補印收據嗎→會當替我補印收據無`
- `可以幫我查醫生門診時間嗎→會當替我查醫生門診時間無`
- `可以幫我查商品規格嗎→會當替我查商品規格無`
- `我想改配送時間→我想欲改配送時間`
- `可以幫我申請退貨嗎→會當替我申請退貨無`
- `可以幫我查門市庫存嗎→會當替我查門市庫存無`
- `可以幫我查轉乘時間嗎→會當替我查轉乘時間無`
- `我想改搭捷運→我想欲改搭捷運`
- `可以幫我查車牌號碼嗎→會當替我查車牌號碼無`

**修正實測錯誤：**
- 飯店、醫療、購物、交通與計程車查詢請託 `可以幫我...` 收斂為 `會當替我...`
- `我想改房型。`、`我想改看診科別。`、`我想改配送時間。`、`我想改搭捷運。` 補足 `想欲`
- 飯店早餐語境固定為 `早頓`，住宿天數固定為 `歇暝天數`
- 計程車 `車牌號碼` 固定長詞組，避免拆成 `車牌號`

**新增迴歸測試：**
- `run_hotel_regression.py / reservation` +2，分類合計 16 筆
- `run_hotel_regression.py / amenities` +2，分類合計 23 筆
- `run_medical_regression.py / registration` +1，分類合計 21 筆
- `run_medical_regression.py / doctor_flow` +1，分類合計 28 筆
- `run_medical_regression.py / pharmacy_payment` +2，分類合計 14 筆
- `run_shopping_regression.py / purchase` +2，分類合計 15 筆
- `run_shopping_regression.py / after_sales` +2，分類合計 30 筆
- `run_transport_regression.py / ticketing` +1，分類合計 17 筆
- `run_transport_regression.py / service_redirect` +1，分類合計 12 筆
- `run_taxi_regression.py / misc` +1，分類合計 15 筆
- hotel 總數 75→79
- medical 總數 91→95
- shopping 總數 80→84
- transport 總數 74→76
- taxi 總數 64→65

**資料同步：**
- 根目錄 `data/` 與 `taigi_converter/data/` 同步更新
- 重編根目錄與 package 內 artifacts

### 2026-05-08：餐廳/家庭/學校/辦公/銀行與公車服務句補強（round217）
**新增人工詞條（13 條 active）：**
- `可以幫我查訂位紀錄嗎→會當替我查訂位紀錄無`
- `可以幫我改訂位時間嗎→會當替我改訂位時間無`
- `我想改用外帶→我想欲改做外帶`
- `可以幫我查藥袋嗎→會當替我查藥袋無`
- `我想改午餐時間→我想欲改晝頓時間`
- `可以幫我提醒媽媽嗎→會當替我提醒阿母無`
- `可以幫我查課表嗎→會當替我查課表無`
- `我想改上課地點→我想欲改上課地點`
- `可以幫我查請假紀錄嗎→會當替我查請假紀錄無`
- `我想改開會地點→我想欲改開會地點`
- `我想改匯款金額→我想欲改匯款金額`
- `可以幫我查公車票價嗎→會當替我查公車票價無`
- `我想改搭公車→我想欲改搭公車`

**修正實測錯誤：**
- 訂位、課務、請假、公車票價等查詢請託 `可以幫我...` 收斂為 `會當替我...`
- `我想改用外帶。`、`我想改午餐時間。`、`我想改上課地點。`、`我想改開會地點。`、`我想改匯款金額。`、`我想改搭公車。` 補足 `想欲`
- 家庭日常 `午餐` 固定為 `晝頓`，親屬稱謂 `媽媽` 固定為 `阿母`
- 餐廳外帶切換使用 `改做外帶`

**新增迴歸測試：**
- `run_restaurant_regression.py / ordering` +3，分類合計 19 筆
- `run_family_regression.py / health_care` +1，分類合計 14 筆
- `run_family_regression.py / daily` +2，分類合計 19 筆
- `run_school_regression.py / student_class` +2，分類合計 18 筆
- `run_workplace_regression.py / meeting` +1，分類合計 19 筆
- `run_workplace_regression.py / leave_availability` +1，分類合計 16 筆
- `run_bank_regression.py / bank_transaction` +1，分類合計 20 筆
- `run_bus_regression.py / payment_cards` +1，分類合計 11 筆
- `run_bus_regression.py / route_transfer` +1，分類合計 15 筆
- restaurant 總數 65→68
- family 總數 59→62
- school 總數 63→65
- workplace 總數 64→66
- bank 總數 64→65
- bus 總數 282→284

**資料同步：**
- 根目錄 `data/` 與 `taigi_converter/data/` 同步更新
- 重編根目錄與 package 內 artifacts

### 2026-05-08：飯店/醫療/購物與交通服務請託補強（round216）
**新增人工詞條（12 條 active）：**
- `可以幫我查入住紀錄嗎→會當替我查入住紀錄無`
- `可以幫我換停車位嗎→會當替我換停車位無`
- `我想改早餐時間→我想欲改早頓時間`
- `可以幫我查檢查結果嗎→會當替我查檢查結果無`
- `可以幫我改領藥地點嗎→會當替我改領藥地點無`
- `我想改病房→我想欲改病房`
- `可以幫我查保固期限嗎→會當替我查保固期限無`
- `可以幫我改退貨方式嗎→會當替我改退貨方式無`
- `我想改發票載具→我想欲改發票載具`
- `可以幫我查候車時間嗎→會當替我查候車時間無`
- `可以幫我改目的地嗎→會當替我改目的地無`
- `我想改搭高鐵→我想欲改搭高鐵`

**修正實測錯誤：**
- 飯店/醫療/購物/交通服務請託 `可以幫我...` 收斂為 `會當替我...`
- `我想改早餐時間。`、`我想改病房。`、`我想改發票載具。`、`我想改搭高鐵。` 補足 `想欲`
- 飯店早餐語境固定為 `早頓`
- 計程車目的地與交通候車時間查詢固定長詞組，避免拆成較直的 `幫我`

**新增迴歸測試：**
- `run_hotel_regression.py / check_in` +2，分類合計 11 筆
- `run_hotel_regression.py / amenities` +1，分類合計 21 筆
- `run_medical_regression.py / tests` +1，分類合計 12 筆
- `run_medical_regression.py / pharmacy_payment` +1，分類合計 12 筆
- `run_medical_regression.py / rooms_inpatient` +1，分類合計 13 筆
- `run_shopping_regression.py / payment` +1，分類合計 13 筆
- `run_shopping_regression.py / after_sales` +2，分類合計 28 筆
- `run_transport_regression.py / ticketing` +2，分類合計 16 筆
- `run_taxi_regression.py / destination` +1，分類合計 16 筆
- hotel 總數 72→75
- medical 總數 88→91
- shopping 總數 77→80
- transport 總數 72→74
- taxi 總數 63→64

**資料同步：**
- 根目錄 `data/` 與 `taigi_converter/data/` 同步更新
- 重編根目錄與 package 內 artifacts

### 2026-05-08：學校/辦公與銀行郵局服務請託補強（round215）
**新增人工詞條（11 條 active）：**
- `可以幫我查成績嗎→會當替我查成績無`
- `我想改考試時間→我想欲改考試時間`
- `可以幫我補交作業嗎→會當替我補交作業無`
- `可以幫我查缺課紀錄嗎→會當替我查缺課紀錄無`
- `可以幫我更新會議連結嗎→會當替我更新會議連結無`
- `我想改報告期限→我想欲改報告期限`
- `可以幫我寄會議紀錄嗎→會當替我寄會議紀錄無`
- `可以幫我查郵局營業時間嗎→會當替我查郵局營業時間無`
- `可以幫我補寄帳單嗎→會當替我補寄帳單無`
- `我想改扣款帳戶→我想欲改扣款口座`
- `可以幫我查轉帳紀錄嗎→會當替我查轉帳紀錄無`

**修正實測錯誤：**
- 學校/辦公/銀行郵局服務請託 `可以幫我...` 收斂為 `會當替我...`
- `我想改考試時間。`、`我想改報告期限。`、`我想改扣款帳戶。` 補足 `想欲`
- 銀行帳戶語境持續使用 `口座`
- 郵局營業時間、會議紀錄、缺課紀錄等長詞組固定，避免拆成較直的 `幫我`

**新增迴歸測試：**
- `run_school_regression.py / student_class` +1，分類合計 16 筆
- `run_school_regression.py / homework` +1，分類合計 10 筆
- `run_school_regression.py / exam` +2，分類合計 10 筆
- `run_workplace_regression.py / meeting` +2，分類合計 18 筆
- `run_workplace_regression.py / workflow` +1，分類合計 17 筆
- `run_bank_regression.py / bank_transaction` +3，分類合計 19 筆
- `run_bank_regression.py / postal` +1，分類合計 15 筆
- school 總數 59→63
- workplace 總數 61→64
- bank 總數 60→64

**資料同步：**
- 根目錄 `data/` 與 `taigi_converter/data/` 同步更新
- 重編根目錄與 package 內 artifacts

### 2026-05-08：家庭/購物/計程車與日常服務請託補強（round214）
**新增人工詞條（11 條 active）：**
- `可以幫我買藥嗎→會當替我買藥仔無`
- `我想改晚餐時間→我想欲改暗頓時間`
- `可以幫我查包裹嗎→會當替我查包裹無`
- `可以幫我改送貨地址嗎→會當替我改送貨地址無`
- `我想改收貨時間→我想欲改收貨時間`
- `可以幫我查發票號碼嗎→會當替我查發票號碼無`
- `可以幫我查司機電話嗎→會當替我查司機電話無`
- `可以幫我改叫車時間嗎→會當替我改叫車時間無`
- `我想改接送時間→我想欲改接送時間`
- `可以幫我查垃圾車時間嗎→會當替我查糞埽車時間無`
- `可以幫我倒垃圾嗎→會當替我摒糞埽無`

**修正實測錯誤：**
- 家庭/購物/計程車服務請託 `可以幫我...` 收斂為 `會當替我...`
- `我想改晚餐時間。`、`我想改收貨時間。`、`我想改接送時間。` 補足 `想欲`
- 家庭日常 `晚餐` 統一收斂為 `暗頓`
- 垃圾車/倒垃圾語境固定為 `糞埽車`、`摒糞埽`

**新增迴歸測試：**
- `run_family_regression.py / health_care` +1，分類合計 13 筆
- `run_family_regression.py / daily` +3，分類合計 17 筆
- `run_shopping_regression.py / payment` +1，分類合計 12 筆
- `run_shopping_regression.py / after_sales` +3，分類合計 26 筆
- `run_taxi_regression.py / hailing` +1，分類合計 13 筆
- `run_taxi_regression.py / misc` +1，分類合計 14 筆
- `run_conversation_regression.py / schedule_plans` +1，分類合計 20 筆
- family 總數 55→59
- shopping 總數 73→77
- taxi 總數 61→63
- conversation 總數 54→55

**資料同步：**
- 根目錄 `data/` 與 `taigi_converter/data/` 同步更新
- 重編根目錄與 package 內 artifacts

### 2026-05-08：飯店/醫療/餐廳與交通服務請託補強（round213）
**新增人工詞條（11 條 active）：**
- `可以幫我查房價嗎→會當替我查房價無`
- `可以幫我補牙刷嗎→會當替我補齒抿仔無`
- `可以幫我換浴巾嗎→會當替我換浴巾無`
- `我想改退房時間→我想欲改退房時間`
- `可以幫我查床位嗎→會當替我查床位無`
- `可以幫我安排檢查時間嗎→會當替我安排檢查時間無`
- `可以幫我查候補名單嗎→會當替我查候補名單無`
- `我想改取藥時間→我想欲改取藥時間`
- `可以幫我換小碗嗎→會當幫我換細碗無`
- `我想改用信用卡付款→我想欲改用信用卡付款`
- `可以幫我查班機時間嗎→會當替我查班機時間無`

**修正實測錯誤：**
- 飯店/醫療/交通查詢請託 `可以幫我查/安排...` 收斂為 `會當替我...`
- `我想改退房時間。`、`我想改取藥時間。`、`我想改用信用卡付款。` 補足 `想欲`
- 飯店 `牙刷` 統一收斂為既有人工用語 `齒抿仔`
- 餐廳 `小碗` 收斂為 `細碗`，避免拆成較不自然的 `碗頭仔`

**新增迴歸測試：**
- `run_hotel_regression.py / reservation` +1，分類合計 14 筆
- `run_hotel_regression.py / check_out` +1，分類合計 13 筆
- `run_hotel_regression.py / amenities` +2，分類合計 20 筆
- `run_medical_regression.py / registration` +1，分類合計 20 筆
- `run_medical_regression.py / tests` +1，分類合計 11 筆
- `run_medical_regression.py / pharmacy_payment` +1，分類合計 11 筆
- `run_medical_regression.py / rooms_inpatient` +1，分類合計 12 筆
- `run_restaurant_regression.py / payment` +1，分類合計 10 筆
- `run_restaurant_regression.py / service` +1，分類合計 17 筆
- `run_transport_regression.py / ticketing` +1，分類合計 14 筆
- hotel 總數 68→72
- medical 總數 84→88
- restaurant 總數 63→65
- transport 總數 71→72

**資料同步：**
- 根目錄 `data/` 與 `taigi_converter/data/` 同步更新
- 重編根目錄與 package 內 artifacts

### 2026-05-08：辦公/學校/銀行帳戶服務請託補強（round212）
**新增人工詞條（10 條 active）：**
- `可以幫我列印文件嗎→會當替我列印文件無`
- `可以幫我通知家長嗎→會當替我通知家長無`
- `可以幫我請事假嗎→會當替我請事假無`
- `可以幫我登記請假嗎→會當替我登記請假無`
- `我想改班表→我想欲改班表`
- `我想改上班時間→我想欲改上班時間`
- `可以幫我改密碼嗎→會當替我改密碼無`
- `可以幫我重設密碼嗎→會當替我重設密碼無`
- `可以幫我更新資料嗎→會當替我更新資料無`
- `我想改聯絡電話→我想欲改聯絡電話`

**修正實測錯誤：**
- 辦公/學校/銀行帳戶服務請託 `可以幫我...` 收斂為 `會當替我...`
- `我想改班表。`、`我想改上班時間。`、`我想改聯絡電話。` 補足 `想欲`
- 密碼重設、資料更新、列印文件等常見服務句用完整 phrase 固定，避免拆成較直的 `幫我`

**新增迴歸測試：**
- `run_bank_regression.py / bank_account` +3，分類合計 18 筆
- `run_bank_regression.py / bank_service` +1，分類合計 12 筆
- `run_school_regression.py / teacher_class` +1，分類合計 13 筆
- `run_school_regression.py / student_class` +1，分類合計 15 筆
- `run_workplace_regression.py / workflow` +1，分類合計 16 筆
- `run_workplace_regression.py / leave_availability` +3，分類合計 15 筆
- bank 總數 56→60
- school 總數 57→59
- workplace 總數 57→61

**資料同步：**
- 根目錄 `data/` 與 `taigi_converter/data/` 同步更新
- 重編根目錄與 package 內 artifacts

### 2026-05-08：預約變更與付款憑證服務請託補強（round211）
**新增人工詞條（7 條 active）：**
- `可以幫我改預約時間嗎→會當替我改預約時間無`
- `可以幫我改預約日期嗎→會當替我改預約日期無`
- `我想改預約日期→我想欲改預約日期`
- `可以幫我取消預約嗎→會當替我取消預約無`
- `可以幫我查訂單嗎→會當替我查訂單無`
- `可以幫我印收據嗎→會當替我印收據無`
- `可以幫我開發票嗎→會當替我開發票無`

**修正實測錯誤：**
- 泛服務請託 `可以幫我改/取消/查...` 收斂為 `會當替我...`
- `我想改預約日期。` 補足 `想欲`
- 付款憑證服務 `印收據`、`開發票` 使用完整 phrase 固定，避免拆成較直的 `幫我`

**新增迴歸測試：**
- `run_medical_regression.py / registration` +1，分類合計 19 筆
- `run_conversation_regression.py / schedule_plans` +1，分類合計 19 筆
- `run_restaurant_regression.py / payment` +1，分類合計 9 筆
- `run_shopping_regression.py / after_sales` +1，分類合計 23 筆
- `run_hotel_regression.py / reservation` +2，分類合計 13 筆
- `run_hotel_regression.py / check_out` +1，分類合計 12 筆
- medical 總數 83→84
- conversation 總數 53→54
- restaurant 總數 62→63
- shopping 總數 72→73
- hotel 總數 65→68

**資料同步：**
- 根目錄 `data/` 與 `taigi_converter/data/` 同步更新
- 重編根目錄與 package 內 artifacts

### 2026-05-07：交通查詢/餐廳服務與日常購物補強（round210）
**新增人工詞條（15 條 active）：**
- `可以幫我查發車時間嗎→會當替我查開車時間無`
- `可以幫我查月台嗎→會當替我查月台無`
- `我想改搭下一班車→我想欲改搭後一班車`
- `可以幫我找服務台嗎→會當替我揣服務台無`
- `可以幫我取消訂位嗎→會當替我取消訂位無`
- `我想改成內用→我想欲改做內用`
- `可以幫我加醬嗎→會當幫我添醬無`
- `可以幫我拿衛生紙嗎→會當幫我提衛生紙來無`
- `我晚點到你家→我較晏到你兜`
- `我等等再出門→我等陣仔閣出門`
- `我晚點再打給你→我較晏閣敲予你`
- `可以幫我查會員資料嗎→會當替我查會員資料無`
- `我想改取貨門市→我想欲改取貨門市`
- `可以幫我取消出貨嗎→會當替我取消出貨無`
- `這個可以幫我退刷嗎→這个會當替我退刷無`

**修正實測錯誤：**
- 交通/購物查詢請託 `可以幫我查...` 收斂為 `會當替我查...`
- `我想改搭下一班車。`、`我想改成內用。`、`我想改取貨門市。` 補足 `想欲`
- 日常 `晚點` 句不再誤轉成等待語境 `等陣仔`
- 餐廳 `加醬`、`拿衛生紙` 收斂為服務場景的 `添...`、`提...來`
- 購物退刷/取消出貨用完整句固定，避免拆成不自然請託

**新增迴歸測試：**
- `run_transport_regression.py / ticketing` +2，分類合計 13 筆
- `run_transport_regression.py / service_redirect` +2，分類合計 11 筆
- `run_restaurant_regression.py / ordering` +2，分類合計 16 筆
- `run_restaurant_regression.py / service` +2，分類合計 16 筆
- `run_conversation_regression.py / schedule_plans` +3，分類合計 18 筆
- `run_shopping_regression.py / payment` +1，分類合計 11 筆
- `run_shopping_regression.py / after_sales` +3，分類合計 22 筆
- transport 總數 67→71
- restaurant 總數 58→62
- conversation 總數 50→53
- shopping 總數 68→72

**資料同步：**
- 根目錄 `data/` 與 `taigi_converter/data/` 同步更新
- 重編根目錄與 package 內 artifacts

### 2026-05-07：醫療/飯店/計程車與辦公請託補強（round209）
**新增人工詞條（16 條 active）：**
- `可以幫我改掛號科別嗎→會當替我改掛號科別無`
- `可以幫我查門診進度嗎→會當替我查門診進度無`
- `我想取消回診→我想欲取消回診`
- `可以幫我申請病歷嗎→會當替我申請病歷無`
- `可以幫我加早餐嗎→會當替我加早頓無`
- `可以幫我換房型嗎→會當替我換房型無`
- `我想改入住時間→我想欲改入住時間`
- `可以幫我叫清潔人員嗎→會當替我叫清潔人員無`
- `可以幫我改上車地點嗎→會當替我改上車地點無`
- `我想改下車地點→我想欲改落車地點`
- `請幫我開窗戶→請替我開窗仔門`
- `可以幫我查車資嗎→會當替我查車錢無`
- `可以幫我預約會議室嗎→會當替我預約會議室無`
- `我想改開會時間→我想欲改開會時間`
- `請幫我追一下進度→請替我追一下進度`
- `可以幫我請公假嗎→會當替我請公假無`

**修正實測錯誤：**
- 醫療/飯店/計程車/辦公服務請託 `可以幫我...` 收斂為 `會當替我...`
- `我想取消回診。`、`我想改入住時間。`、`我想改下車地點。` 補足 `想欲`
- `我想改開會時間。` 不再把 `開會時間` 誤切成錯誤輸出
- 飯店 `早餐` 統一收斂為 `早頓`
- 計程車 `上車地點`、`車資` 用完整 phrase 固定，避免拆成不穩定字詞

**新增迴歸測試：**
- `run_medical_regression.py / registration` +1，分類合計 18 筆
- `run_medical_regression.py / doctor_flow` +2，分類合計 27 筆
- `run_medical_regression.py / redirect` +1，分類合計 7 筆
- `run_hotel_regression.py / reservation` +1，分類合計 11 筆
- `run_hotel_regression.py / amenities` +1，分類合計 18 筆
- `run_hotel_regression.py / issues` +2，分類合計 16 筆
- `run_taxi_regression.py / destination` +2，分類合計 15 筆
- `run_taxi_regression.py / payment` +1，分類合計 9 筆
- `run_taxi_regression.py / misc` +1，分類合計 13 筆
- `run_workplace_regression.py / meeting` +2，分類合計 16 筆
- `run_workplace_regression.py / workflow` +1，分類合計 15 筆
- `run_workplace_regression.py / leave_availability` +1，分類合計 12 筆
- medical 總數 79→83
- hotel 總數 61→65
- taxi 總數 57→61
- workplace 總數 53→57

**資料同步：**
- 根目錄 `data/` 與 `taigi_converter/data/` 同步更新
- 重編根目錄與 package 內 artifacts

### 2026-05-07：銀行查詢/購物售後與學校家庭請託補強（round208）
**新增人工詞條（10 條 active）：**
- `可以幫我查匯率嗎→會當替我查匯率無`
- `可以幫我查貸款利率嗎→會當替我查貸款利率無`
- `可以幫我開存款證明嗎→會當替我開存款證明無`
- `可以幫我查配送狀態嗎→會當替我查配送狀態無`
- `可以幫我換顏色嗎→會當替我換顏色無`
- `我想改付款方式→我想欲改付款方式`
- `我想跟老師請假→我想欲佮老師請假`
- `可以幫我聯絡導師嗎→會當替我聯絡導師無`
- `我晚點去接小孩→我較晏去接囡仔`
- `可以幫我煮晚餐嗎→會當替我煮暗頓無`

**修正實測錯誤：**
- 銀行/購物/學校服務請託 `可以幫我...` 收斂為 `會當替我...`
- `我想改付款方式。`、`我想跟老師請假。` 補足 `想欲`
- `我晚點去接小孩。` 不再把 `晚點` 誤轉成等待語境 `等陣仔`
- 家庭晚餐語境統一使用 `暗頓`

**新增迴歸測試：**
- `run_bank_regression.py / bank_account` +1，分類合計 15 筆
- `run_bank_regression.py / bank_transaction` +1，分類合計 16 筆
- `run_bank_regression.py / bank_service` +1，分類合計 11 筆
- `run_shopping_regression.py / purchase` +1，分類合計 13 筆
- `run_shopping_regression.py / payment` +1，分類合計 10 筆
- `run_shopping_regression.py / after_sales` +1，分類合計 19 筆
- `run_school_regression.py / teacher_class` +1，分類合計 12 筆
- `run_school_regression.py / student_class` +1，分類合計 14 筆
- `run_family_regression.py / parent_child` +1，分類合計 12 筆
- `run_family_regression.py / daily` +1，分類合計 14 筆
- bank 總數 53→56
- shopping 總數 65→68
- school 總數 55→57
- family 總數 53→55

**資料同步：**
- 根目錄 `data/` 與 `taigi_converter/data/` 同步更新
- 重編根目錄與 package 內 artifacts

### 2026-05-07：交通查詢/餐廳客製與日常訊息補強（round207）
**新增人工詞條（8 條 active）：**
- `可以幫我查車票價格嗎→會當替我查車票價錢無`
- `可以幫我查車班嗎→會當替我查車班無`
- `這班車會停哪幾站→這班車會停佗幾站`
- `可以幫我換成外帶嗎→會當替我換做外帶無`
- `這個可以不要辣嗎→這个會當免辣無`
- `可以幫我拿吸管嗎→會當幫我提吸管來無`
- `我快到了→我欲到矣`
- `我晚點再跟你說→我較晏閣共你講`

**修正實測錯誤：**
- 交通查詢 `可以幫我查...` 收斂為 `會當替我查...`
- `這班車會停哪幾站？` 補足 `哪幾站→佗幾站`
- 餐廳外帶/客製句避免 `換成`、`不要辣` 的華語式直譯
- `我快到了。` 不再誤作速度副詞 `緊到矣`
- `我晚點再跟你說。` 不再把 `晚點` 誤轉成等待語境 `等陣仔`

**新增迴歸測試：**
- `run_transport_regression.py / ticketing` +2，分類合計 11 筆
- `run_transport_regression.py / service_redirect` +1，分類合計 9 筆
- `run_restaurant_regression.py / ordering` +1，分類合計 14 筆
- `run_restaurant_regression.py / spice_dietary` +1，分類合計 10 筆
- `run_restaurant_regression.py / service` +1，分類合計 14 筆
- `run_conversation_regression.py / status_check` +1，分類合計 7 筆
- `run_conversation_regression.py / schedule_plans` +1，分類合計 15 筆
- transport 總數 64→67
- restaurant 總數 55→58
- conversation 總數 48→50

**資料同步：**
- 根目錄 `data/` 與 `taigi_converter/data/` 同步更新
- 重編根目錄與 package 內 artifacts

### 2026-05-07：家庭早餐/醫療掛號與飯店房務補強（round206）
**新增人工詞條（8 條 active）：**
- `可以幫我買早餐嗎→會當替我買早頓無`
- `我晚點帶小孩回家→我較晏帶囡仔轉去厝裡`
- `可以幫我掛急診嗎→會當替我掛急診無`
- `我想改回診時間→我想欲改回診時間`
- `可以幫我查藥單嗎→會當替我查藥單無`
- `可以幫我叫客房服務嗎→會當替我叫客房服務無`
- `房間可以晚一點打掃嗎→房間會當較晏一點拚掃無`
- `可以幫我換房卡嗎→會當替我換房卡無`

**修正實測錯誤：**
- 家庭/醫療/飯店服務請託 `可以幫我...` 收斂為 `會當替我...`
- `我晚點帶小孩回家。` 不再把 `晚點` 誤轉成等待語境 `等陣仔`，也不再把 `帶小孩` 誤轉成 `夾細囝`
- `我想改回診時間。` 補足 `想欲`
- `房間可以晚一點打掃嗎？` 收斂為飯店房務語境 `較晏一點拚掃`

**新增迴歸測試：**
- `run_family_regression.py / parent_child` +1，分類合計 11 筆
- `run_family_regression.py / daily` +1，分類合計 13 筆
- `run_medical_regression.py / registration` +1，分類合計 17 筆
- `run_medical_regression.py / doctor_flow` +1，分類合計 25 筆
- `run_medical_regression.py / pharmacy_payment` +1，分類合計 10 筆
- `run_hotel_regression.py / amenities` +2，分類合計 17 筆
- `run_hotel_regression.py / issues` +1，分類合計 14 筆
- family 總數 51→53
- medical 總數 76→79
- hotel 總數 58→61

**資料同步：**
- 根目錄 `data/` 與 `taigi_converter/data/` 同步更新
- 重編根目錄與 package 內 artifacts

### 2026-05-07：購物出貨/學校影印與辦公會議補強（round205）
**新增人工詞條（8 條 active）：**
- `可以幫我查出貨進度嗎→會當替我查出貨進度無`
- `我想改收件地址→我想欲改收件地址`
- `這件可以幫我留一下嗎→這件會當替我留一下無`
- `可以幫我包起來嗎→會當替我包起來無`
- `可以幫我影印講義嗎→會當替我影印講義無`
- `我想改會議地點→我想欲改會議地點`
- `可以幫我安排會議室嗎→會當替我安排會議室無`
- `我晚點再回覆你→我較晏閣回覆你`

**修正實測錯誤：**
- 購物/學校/辦公服務請託 `可以幫我...` 收斂為 `會當替我...`
- `我想改收件地址。`、`我想改會議地點。` 補足 `想欲`
- `我晚點再回覆你。` 不再把 `晚點` 誤轉成等待語境 `等陣仔`
- 購物留貨/包裝句用長 phrase 固定，避免後續拆詞造成語氣不自然

**新增迴歸測試：**
- `run_shopping_regression.py / purchase` +2，分類合計 12 筆
- `run_shopping_regression.py / after_sales` +2，分類合計 18 筆
- `run_school_regression.py / teacher_class` +1，分類合計 11 筆
- `run_workplace_regression.py / meeting` +2，分類合計 14 筆
- `run_conversation_regression.py / schedule_plans` +1，分類合計 14 筆
- shopping 總數 61→65
- school 總數 54→55
- workplace 總數 51→53
- conversation 總數 47→48

**資料同步：**
- 根目錄 `data/` 與 `taigi_converter/data/` 同步更新
- 重編根目錄與 package 內 artifacts

### 2026-05-07：日常通話/計程車路線與銀行帳單補強（round204）
**新增人工詞條（8 條 active）：**
- `你方便說話嗎→你方便講話無`
- `晚點再聊→較晏閣聊`
- `可以幫我叫大車嗎→會當替我叫大車無`
- `可以幫我看一下路線嗎→會當替我看覓路線無`
- `我想先去便利商店→我想欲先去便利店`
- `我想查信用卡帳單→我想欲查信用卡費用明細`
- `可以幫我補辦提款卡嗎→會當替我補辦提款卡無`
- `可以幫我寄到國外嗎→會當替我寄到國外無`

**修正實測錯誤：**
- `晚點再聊。` 不再把 `晚點` 誤轉成等待語境 `等陣仔`
- 計程車/銀行/郵局服務請託 `可以幫我...` 收斂為 `會當替我...`
- `我想先去便利商店。` 補足 `想欲` 並與既有 taxi 便利店用語一致
- `我想查信用卡帳單。` 避開舊 base `帳單→數單` 污染，改收斂為 `信用卡費用明細`

**新增迴歸測試：**
- `run_conversation_regression.py / daily_response` +1，分類合計 16 筆
- `run_conversation_regression.py / schedule_plans` +1，分類合計 13 筆
- `run_taxi_regression.py / hailing` +1，分類合計 12 筆
- `run_taxi_regression.py / destination` +1，分類合計 13 筆
- `run_taxi_regression.py / misc` +1，分類合計 12 筆
- `run_bank_regression.py / bank_account` +1，分類合計 14 筆
- `run_bank_regression.py / bank_transaction` +1，分類合計 15 筆
- `run_bank_regression.py / postal` +1，分類合計 14 筆
- conversation 總數 45→47
- taxi 總數 54→57
- bank 總數 50→53

**資料同步：**
- 根目錄 `data/` 與 `taigi_converter/data/` 同步更新
- 重編根目錄與 package 內 artifacts

### 2026-05-07：交通查詢與餐廳服務請託補強（round203）
**新增人工詞條（6 條 active）：**
- `可以幫我查末班車時間嗎→會當替我查尾班車時間無`
- `可以幫我查轉乘路線嗎→會當替我查轉乘路線無`
- `可以幫我加湯嗎→會當幫我添湯無`
- `可以幫我換座位嗎→會當替我換座位無`
- `可以不要放蔥嗎→會當免放蔥無`
- `這道菜可以快一點嗎→這道菜會當較緊無`

**修正實測錯誤：**
- 交通服務查詢 `可以幫我查...` 收斂為 `會當替我查...`
- 餐廳 `加湯` 收斂為較自然的 `添湯`
- `這道菜可以快一點嗎？` 不再多出不自然的 `咧無`
- `可以不要放蔥嗎？` 收斂為餐點客製語境 `免放蔥`

**新增迴歸測試：**
- `run_transport_regression.py / ticketing` +1，分類合計 9 筆
- `run_transport_regression.py / service_redirect` +1，分類合計 8 筆
- `run_restaurant_regression.py / spice_dietary` +1，分類合計 9 筆
- `run_restaurant_regression.py / seating` +1，分類合計 12 筆
- `run_restaurant_regression.py / service` +2，分類合計 13 筆
- transport 總數 62→64
- restaurant 總數 51→55

**資料同步：**
- 根目錄 `data/` 與 `taigi_converter/data/` 同步更新
- 重編根目錄與 package 內 artifacts

### 2026-05-07：飯店服務請託與醫療報告/領藥補強（round202）
**新增人工詞條（8 條 active）：**
- `可以幫我補毛巾嗎→會當替我補面巾無`
- `可以幫我修冷氣嗎→會當替我修空調無`
- `房間可以不要打掃嗎→房間會當免拚掃無`
- `可以幫我換枕頭嗎→會當替我換枕頭無`
- `我想查檢查報告→我想欲查檢查報告`
- `可以幫我查報告嗎→會當替我查報告無`
- `我要領慢性病藥→我欲領慢性病藥`
- `可以幫我量血壓嗎→會當替我量血壓無`

**修正實測錯誤：**
- 飯店/醫療服務請託 `可以幫我...` 收斂為 `會當替我...`
- `我要領慢性病藥。` 不再把 `慢性病` 誤拆成怪字
- `房間可以不要打掃嗎？` 收斂為飯店房務語境 `免拚掃`
- 醫療查報告句補足 `想欲`

**新增迴歸測試：**
- `run_hotel_regression.py / amenities` +2，分類合計 15 筆
- `run_hotel_regression.py / issues` +2，分類合計 13 筆
- `run_medical_regression.py / doctor_flow` +2，分類合計 24 筆
- `run_medical_regression.py / pharmacy_payment` +2，分類合計 9 筆
- hotel 總數 54→58
- medical 總數 72→76

**資料同步：**
- 根目錄 `data/` 與 `taigi_converter/data/` 同步更新
- 重編根目錄與 package 內 artifacts

### 2026-05-07：銀行明細/購物庫存與學校請假補強（round201）
**新增人工詞條（9 條 active）：**
- `我想查帳戶明細→我想欲查口座明細`
- `可以幫我列印明細嗎→會當替我列印明細無`
- `我想改通訊地址→我想欲改通訊地址`
- `可以幫我查庫存嗎→會當替我查庫存無`
- `可以幫我換尺寸嗎→會當替我換尺寸無`
- `發票可以重開嗎→發票會當重開無`
- `作業可以晚一點交嗎→作業會當較晏一點交無`
- `我想晚一點到學校→我想欲較晏一點到學校`
- `可以幫我請假嗎→會當替我請假無`

**修正實測錯誤：**
- `我想查帳戶明細。` 不再把 `帳戶` 誤拆成 `數戶`
- 銀行/購物/學校服務請託 `可以幫我...` 收斂為 `會當替我...`
- `作業可以晚一點交嗎？` 不再把 `晚一點` 誤轉成等待語境 `等陣仔`
- `我想晚一點到學校。` 補足 `想欲`

**新增迴歸測試：**
- `run_bank_regression.py / bank_account` +1，分類合計 13 筆
- `run_bank_regression.py / bank_transaction` +2，分類合計 14 筆
- `run_shopping_regression.py / purchase` +2，分類合計 10 筆
- `run_shopping_regression.py / payment` +1，分類合計 9 筆
- `run_school_regression.py / student_class` +2，分類合計 13 筆
- `run_school_regression.py / homework` +1，分類合計 9 筆
- bank 總數 47→50
- shopping 總數 58→61
- school 總數 51→54

**資料同步：**
- 根目錄 `data/` 與 `taigi_converter/data/` 同步更新
- 重編根目錄與 package 內 artifacts

### 2026-05-07：辦公晚點/家庭照護日常 phrase 補強（round200）
**新增人工詞條（10 條 active）：**
- `我晚點進公司→我較晏進公司`
- `資料我晚點補→資料我較晏補`
- `請幫我看一下這份資料→請替我看覓這份資料`
- `可以幫我代班嗎→會當替我代班無`
- `可以幫我改時間嗎→會當替我改時間無`
- `我帶小孩去學校→我帶囡仔去學校`
- `可以幫我接小孩嗎→會當替我接囡仔無`
- `小孩肚子痛→囡仔腹肚疼`
- `晚餐要吃什麼→暗頓欲食啥`
- `晚餐吃什麼→暗頓食啥`

**修正實測錯誤：**
- `我晚點進公司。`、`資料我晚點補。` 不再把 `晚點` 誤轉成等待語境 `等陣仔`
- 辦公請託 `可以幫我...` 收斂為 `會當替我...`
- `我帶小孩去學校。` 不再把 `帶小孩` 誤轉成 `夾細囝`
- `小孩肚子痛。` 收斂為身體部位疼痛語境 `腹肚疼`
- 家庭日常 `晚餐` 統一收斂為 `暗頓`

**新增迴歸測試：**
- `run_workplace_regression.py / workflow` +2，分類合計 14 筆
- `run_workplace_regression.py / leave_availability` +3，分類合計 11 筆
- `run_family_regression.py / parent_child` +2，分類合計 10 筆
- `run_family_regression.py / health_care` +1，分類合計 12 筆
- `run_family_regression.py / daily` +2，分類合計 12 筆
- workplace 總數 46→51
- family 總數 46→51

**資料同步：**
- 根目錄 `data/` 與 `taigi_converter/data/` 同步更新
- 重編根目錄與 package 內 artifacts

### 2026-05-07：日常訊息與餐廳服務 phrase 補強（round199）
**停用舊人工詞條（1 條）：**
- `路上小心→路裡小心`：保留歷史但停用，避免 `小心` 保留華語字面

**新增人工詞條（8 條 active，淨增 7 條 active）：**
- `路上小心→路裡細膩`
- `我晚點回你→我較晏閣回你`
- `我到了再打給你→我到矣閣敲予你`
- `餐點可以快一點嗎→餐點會當較緊無`
- `可以幫我加飯嗎→會當幫我添飯無`
- `可以幫我拿餐具嗎→會當幫我提餐具來無`
- `我想取消訂位→我想欲取消訂位`
- `我想改訂位時間→我想欲改訂位時間`

**修正實測錯誤：**
- `路上小心。` 從 `路裡小心` 收斂為較自然的 `路裡細膩`
- `我晚點回你。` 不再把 `晚點` 誤轉成等待語境 `等陣仔`
- `餐點可以快一點嗎？` 不再多出不自然的 `咧無`
- 餐廳服務請託補足 `添飯`、`提餐具來` 這類動作語境
- 訂位異動句補足 `想欲`

**新增迴歸測試：**
- `run_conversation_regression.py / schedule_plans` +3，分類合計 12 筆
- `run_restaurant_regression.py / ordering` +2，分類合計 13 筆
- `run_restaurant_regression.py / service` +3，分類合計 11 筆
- conversation 總數 42→45
- restaurant 總數 46→51

**資料同步：**
- 根目錄 `data/` 與 `taigi_converter/data/` 同步更新
- 重編根目錄與 package 內 artifacts

### 2026-05-07：計程車叫車/目的地與付款請託補強（round198）
**新增人工詞條（7 條 active）：**
- `可以幫我開後車廂嗎→會當替我開後行李箱無`
- `我想晚一點出發→我想欲較晏一點出發`
- `到機場要多久→到機場愛偌久`
- `我想先去加油站→我想欲先去加油站`
- `請幫我等一下→請替我等一下仔`
- `我想改成現金付款→我想欲改做現錢付款`
- `可以幫我聯絡司機嗎→會當替我聯絡司機無`

**修正實測錯誤：**
- `可以幫我...` 計程車服務請託收斂為 `會當替我...`
- `我想晚一點出發。` 不再把 `晚一點` 誤轉成等待語境 `等陣仔`
- `到機場要多久？` 收斂為目的地時間需求語境 `愛偌久`
- `我想改成現金付款。` 補足 `想欲` 與 `改做`

**新增迴歸測試：**
- `run_taxi_regression.py / hailing` +1，分類合計 11 筆
- `run_taxi_regression.py / destination` +2，分類合計 12 筆
- `run_taxi_regression.py / payment` +1，分類合計 8 筆
- `run_taxi_regression.py / misc` +3，分類合計 11 筆
- taxi 總數 47→54

**資料同步：**
- 根目錄 `data/` 與 `taigi_converter/data/` 同步更新
- 重編根目錄與 package 內 artifacts

### 2026-05-07：飯店退房/設施與房間問題補強（round197）
**新增人工詞條（7 條 active）：**
- `我想晚一點退房→我想欲較晏一點退房`
- `可以幫我保管行李嗎→會當替我保管行李無`
- `可以換安靜一點的房間嗎→會當換較恬的房間無`
- `早餐幾點開始→早頓幾點開始`
- `房間要打掃嗎→房間愛拚掃無`
- `我想加一條棉被→我想欲加一領棉被`
- `可以幫我叫計程車嗎→會當替我叫計程車無`

**修正實測錯誤：**
- `我想晚一點退房。` 不再把 `晚一點` 誤轉成等待語境 `等陣仔`
- 飯店服務請託 `可以幫我...` 收斂為 `會當替我...`
- `房間要打掃嗎？` 收斂為房務義務語境 `愛`
- `可以換安靜一點的房間嗎？` 收斂為較自然的 `較恬的房間`

**新增迴歸測試：**
- `run_hotel_regression.py / check_in` +1，分類合計 9 筆
- `run_hotel_regression.py / check_out` +2，分類合計 11 筆
- `run_hotel_regression.py / amenities` +3，分類合計 13 筆
- `run_hotel_regression.py / issues` +1，分類合計 11 筆
- hotel 總數 47→54

**資料同步：**
- 根目錄 `data/` 與 `taigi_converter/data/` 同步更新
- 重編根目錄與 package 內 artifacts

### 2026-05-07：銀行提款卡/郵件查詢情境補強（round196）
**新增人工詞條（6 條 active）：**
- `我想改提款卡密碼→我想欲改提款卡密碼`
- `提款卡被鎖住了→提款卡鎖牢矣`
- `可以幫我查餘額嗎→會當替我查餘額無`
- `我要取消自動扣款→我欲取消自動扣款`
- `這個包裹要多久會到→這个包裹偌久會到`
- `可以幫我查郵件狀態嗎→會當替我查郵件狀態無`

**修正實測錯誤：**
- `提款卡被鎖住了。` 不再保留華語被字句
- `可以幫我查餘額嗎？`、`可以幫我查郵件狀態嗎？` 收斂為服務請託語境 `替我查`
- `我想改提款卡密碼。` 補足 `想欲`
- `這個包裹要多久會到？` 收斂為 `偌久會到`

**新增迴歸測試：**
- `run_bank_regression.py / bank_account` +2，分類合計 12 筆
- `run_bank_regression.py / bank_transaction` +2，分類合計 12 筆
- `run_bank_regression.py / postal` +2，分類合計 13 筆
- bank 總數 41→47

**資料同步：**
- 根目錄 `data/` 與 `taigi_converter/data/` 同步更新
- 重編根目錄與 package 內 artifacts

### 2026-05-07：家庭照護/日常請託情境補強（round195）
**新增人工詞條（6 條 active）：**
- `媽媽說晚上要吃藥→阿母講暗時愛食藥仔`
- `爸爸明天要回診→阿爸明仔載愛回診`
- `我可以晚一點回家嗎→我會當較晏一點轉去厝裡無`
- `請幫我照顧弟弟→請替我照顧阿弟仔`
- `今天要帶小孩去看醫生→今仔日愛帶囡仔去予醫生看`
- `請幫我買晚餐→請替我買暗頓`

**修正實測錯誤：**
- `我可以晚一點回家嗎？` 不再把 `晚一點` 誤轉成等待語境 `等陣仔`
- `今天要帶小孩去看醫生。` 不再把 `帶小孩` 誤轉成 `夾細囝`
- 家庭請託句 `請幫我...` 收斂為 `請替我...`
- 家庭照護/就醫語境的 `要` 收斂為 `愛`

**新增迴歸測試：**
- `run_family_regression.py / health_care` +3，分類合計 11 筆
- `run_family_regression.py / siblings` +1，分類合計 9 筆
- `run_family_regression.py / daily` +2，分類合計 10 筆
- family 總數 40→46

**資料同步：**
- 根目錄 `data/` 與 `taigi_converter/data/` 同步更新
- 重編根目錄與 package 內 artifacts

### 2026-05-07：辦公會議/工作流程情境補強（round194）
**新增人工詞條（8 條 active）：**
- `我可以晚一點交報告嗎→我會當較晏一點交報告無`
- `報告要在下班前交→報告愛佇下班前交`
- `主管說明天要開會→主管講明仔載欲開會`
- `會議要準備資料嗎→會議愛準備資料無`
- `請幫我確認會議時間→請替我確認會議時間`
- `可以改成線上會議嗎→會當改做線上會議無`
- `請幫我轉給主管→請替我轉予主管`
- `這份資料要更新→這份資料愛更新`

**修正實測錯誤：**
- `請幫我確認會議時間。` 不再把 `會議時間` 誤拆成 `表決間`
- `我可以晚一點交報告嗎？` 不再把 `晚一點` 誤轉成等待語境 `等陣仔`
- `會議要準備資料嗎？`、`這份資料要更新。` 收斂為工作義務語境 `愛`
- `請幫我轉給主管。` 收斂為 `請替我轉予主管`

**新增迴歸測試：**
- `run_workplace_regression.py / meeting` +4，分類合計 12 筆
- `run_workplace_regression.py / workflow` +4，分類合計 12 筆
- workplace 總數 38→46

**資料同步：**
- 根目錄 `data/` 與 `taigi_converter/data/` 同步更新
- 重編根目錄與 package 內 artifacts

### 2026-05-07：學校作業/考試/圖書館情境補強（round193）
**新增人工詞條（5 條 active）：**
- `我可以晚一點交作業嗎→我會當較晏一點交作業無`
- `作業要寫第幾頁→作業愛寫第幾頁`
- `老師說明天要小考→老師講明仔載愛考小考`
- `考試要帶鉛筆嗎→考試愛帶鉛筆無`
- `我想借這本書→我想欲借這本冊`

**修正實測錯誤：**
- `我可以晚一點交作業嗎？` 不再把 `晚一點` 誤轉成等待語境 `等陣仔`
- `作業要寫第幾頁？`、`考試要帶鉛筆嗎？` 收斂為課業義務語境 `愛`
- `老師說明天要小考。` 不再保留 `要小考`
- `我想借這本書。` 補足 `想欲` 與 `書→冊` 的整段轉換

**新增迴歸測試：**
- `run_school_regression.py / homework` +2，分類合計 8 筆
- `run_school_regression.py / exam` +2，分類合計 8 筆
- `run_school_regression.py / campus` +1，分類合計 9 筆
- school 總數 46→51

**資料同步：**
- 根目錄 `data/` 與 `taigi_converter/data/` 同步更新
- 重編根目錄與 package 內 artifacts

### 2026-05-07：購物售後退款/物流細節補強（round192）
**新增人工詞條（6 條 active）：**
- `拆封了還可以換貨嗎→拆封矣猶會當換貨無`
- `可以只退一部分嗎→會當干焦退一部分無`
- `退款要等幾天→退錢愛等幾工`
- `商品少了一個配件→商品欠一个配件`
- `我想改送貨時間→我想欲改送貨時間`
- `物流一直沒有更新→物流攏無更新`

**修正實測錯誤：**
- `拆封了還可以換貨嗎？` 不再保留華語完成標記 `了`
- `可以只退一部分嗎？` 收斂為 `干焦退一部分`
- `退款要等幾天？` 收斂為售後等待語境 `退錢愛等幾工`
- `商品少了一個配件。` 收斂為缺件語境 `欠一个配件`
- `物流一直沒有更新。` 收斂為較自然的 `物流攏無更新`

**新增迴歸測試：**
- `run_shopping_regression.py / after_sales` +6，分類合計 16 筆
- shopping 總數 52→58

**資料同步：**
- 根目錄 `data/` 與 `taigi_converter/data/` 同步更新
- 重編根目錄與 package 內 artifacts

### 2026-05-07：購物議價與學校課堂義務語境補強（round191）
**新增人工詞條（8 條 active）：**
- `買多有折扣嗎→買較濟有拍折無`
- `可以打九折嗎→會當拍九折無`
- `這個有特價嗎→這个有特價無`
- `今天要考小考嗎→今仔日愛考小考無`
- `老師請大家安靜→老師請逐家恬恬`
- `今天要帶課本嗎→今仔日愛帶課本無`
- `下課後要留下來嗎→下課後愛留落來無`
- `明天要帶聯絡簿嗎→明仔載愛帶聯絡簿無`

**修正實測錯誤：**
- `買多有折扣嗎？` 不再保留華語比較量詞 `買多`
- `可以打九折嗎？` 收斂為購物議價語境的 `拍九折`
- `今天要考小考嗎？`、`今天要帶課本嗎？`、`明天要帶聯絡簿嗎？` 收斂為課堂義務語境 `愛`
- `老師請大家安靜。` 收斂為較自然的 `請逐家恬恬`

**新增迴歸測試：**
- `run_shopping_regression.py / bargaining` +3，分類合計 10 筆
- `run_school_regression.py / teacher_class` +3，分類合計 10 筆
- `run_school_regression.py / student_class` +3，分類合計 11 筆
- shopping 總數 49→52；school 總數 40→46

**資料同步：**
- 根目錄 `data/` 與 `taigi_converter/data/` 同步更新
- 重編根目錄與 package 內 artifacts

### 2026-05-07：購物議價/售後與學校班級情境補強（round190）
**新增人工詞條（14 條 active；runtime 13 條 + identity 保護 1 條）：**
- `可以再便宜一點嗎→會當閣較俗無`
- `買兩個可以算便宜一點嗎→買兩个會當算俗淡薄仔無`
- `有送贈品嗎→有送贈品無`
- `我想查退款進度→我想欲查退錢進度`
- `可以換同款不同尺寸嗎→會當換同款無仝尺寸無`
- `這個有保固嗎→這个有保固無`
- `物流延遲了→物流延遲矣`
- `我今天要請假→我今仔日欲請假`
- `老師今天會點名嗎→老師今仔日會點名無`
- `請幫我聯絡家長→請替我聯絡家長`
- `學生今天請假→學生今仔日請假`
- `他今天遲到了→伊今仔日遲到矣`
- `請家長簽聯絡簿→請家長簽聯絡簿`
- `明天要交作業→明仔載愛交作業`

**修正實測錯誤：**
- `我今天要請假。` 不再保留華語義務/意圖未分化的 `要請假`
- `請幫我聯絡家長。` 不再輸出 `請共我聯絡家長`
- `明天要交作業。` 收斂為作業繳交義務語境的 `愛交`
- 購物售後補足退款進度、換尺寸、保固與物流延遲；議價補足再便宜、買多議價與贈品

**新增迴歸測試：**
- `run_shopping_regression.py / bargaining` +3，分類合計 7 筆
- `run_shopping_regression.py / after_sales` +4，分類合計 10 筆
- `run_school_regression.py / teacher_class` +3，分類合計 7 筆
- `run_school_regression.py / student_class` +4，分類合計 8 筆
- shopping 總數 42→49；school 總數 33→40

**資料同步：**
- 根目錄 `data/` 與 `taigi_converter/data/` 同步更新
- 重編根目錄與 package 內 artifacts

### 2026-05-06：餐廳點餐/座位與計程車叫車情境補強（round189）
**新增人工詞條（15 條 active）：**
- `我想加點一份薯條→我想欲加點一份薯條`
- `可以換成套餐嗎→會當換做套餐無`
- `這份可以做外帶嗎→這份會當做外帶無`
- `飲料可以去冰嗎→飲料會當去冰無`
- `可以先上飲料嗎→會當先送飲料來無`
- `可以坐靠窗嗎→會當坐靠窗無`
- `需要等位嗎→需要等位無`
- `可以併桌嗎→會當併桌無`
- `有四個人的位子嗎→有四个人的位子無`
- `可以換到裡面的位置嗎→會當換到內底的位無`
- `請在門口等我→請佇門跤口等我`
- `我們有四個人→阮有四个人`
- `可以派大一點的車嗎→會當派較大的車無`
- `車可以等五分鐘嗎→車會當等五分鐘無`
- `我在便利商店門口等車→我佇便利店門跤口等車`

**修正實測錯誤：**
- `我想加點一份薯條。` 不再整句保留華語
- `這份可以做外帶嗎？` 不再輸出不自然的 `會做得外帶`
- `有四個人的位子嗎？` 收斂為座位語境整句，避免量詞與字形漂移
- `我們有四個人。` 在對司機說明乘客人數時收斂為排除對方的 `阮`
- `我在便利商店門口等車。` 收斂為 `佇便利店門跤口等車`

**新增迴歸測試：**
- `run_restaurant_regression.py / ordering` +5，分類合計 11 筆
- `run_restaurant_regression.py / seating` +5，分類合計 11 筆
- `run_taxi_regression.py / hailing` +5，分類合計 10 筆
- restaurant 總數 36→46；taxi 總數 42→47

**資料同步：**
- 根目錄 `data/` 與 `taigi_converter/data/` 同步更新
- 重編根目錄與 package 內 artifacts

### 2026-05-06：飯店設施/退房與郵局寄件情境補強（round188）
**新增人工詞條（13 條 active）：**
- `請問有健身房嗎→借問有健身房無`
- `請問有洗衣機嗎→借問有洗衣機無`
- `房間有wifi嗎→房間有wifi無`
- `可以多給我一雙拖鞋嗎→會當加予我一雙淺拖仔無`
- `有牙刷嗎→有齒抿仔無`
- `可以晚一點退房嗎→會當較晏一點退房無`
- `房卡要交回哪裡→房卡愛交轉去佗位`
- `可以幫我叫車嗎→會當替我叫車無`
- `我要寄國際包裹→我欲寄國際包裹`
- `我要寄平信→我欲寄平信`
- `郵資多少錢→郵資偌濟錢`
- `郵遞區號要寫嗎→郵遞區號愛寫無`
- `可以查追蹤號碼嗎→會當查追蹤號碼無`

**修正實測錯誤：**
- `可以晚一點退房嗎？` 不再把 `晚一點` 誤轉成等待語境的 `等陣仔`
- `可以多給我一雙拖鞋嗎？` 收斂為飯店備品語境的 `會當加予我一雙淺拖仔無？`
- `房卡要交回哪裡？` 收斂為退房交回語境 `房卡愛交轉去佗位？`
- 郵局寄件情境補足國際包裹、平信、郵資、郵遞區號與追蹤號碼

**新增迴歸測試：**
- `run_hotel_regression.py / amenities` +5，分類合計 10 筆
- `run_hotel_regression.py / check_out` +3，分類合計 9 筆
- `run_bank_regression.py / postal` +5，分類合計 11 筆
- hotel 總數 39→47；bank 總數 36→41

**資料同步：**
- 根目錄 `data/` 與 `taigi_converter/data/` 同步更新
- 重編根目錄與 package 內 artifacts

### 2026-05-06：計程車目的地/導航與飯店問題反映補強（round187）
**新增人工詞條（15 條 active；runtime 12 條 + identity 保護 3 條）：**
- `我想改目的地→我想欲改目的地`
- `請走高速公路→請走高速公路`
- `我要到捷運站→我欲到捷運站`
- `我要到醫院→我欲到病院`
- `請載我到飯店→請載我到飯店`
- `靠右邊停→靠正手爿停`
- `靠左邊停→靠倒手爿停`
- `前面路口右轉→頭前路口正斡`
- `不要走高速公路→莫走高速公路`
- `可以迴轉嗎→會當踅頭無`
- `房間漏水→房間漏水`
- `房間沒有毛巾→房間無面巾`
- `可以幫我換房嗎→會當替我換房無`
- `遙控器壞了→遙控器歹去矣`
- `房間太冷了→房間太寒矣`

**修正實測錯誤：**
- `我想改目的地。` 不再輸出不自然的 `我想改欲去的所在。`
- `可以幫我換房嗎？` 不再保留 `幫我`
- `房間沒有毛巾。` 收斂為 `房間無面巾。`
- 計程車目的地/導航類補足改目的地、捷運站、醫院、靠左右、路口右轉與迴轉

**新增迴歸測試：**
- `run_taxi_regression.py / destination` +5，分類合計 10 筆
- `run_taxi_regression.py / navigation` +5，分類合計 12 筆
- `run_hotel_regression.py / issues` +5，分類合計 10 筆
- taxi 總數 32→42；hotel 總數 34→39

**資料同步：**
- 根目錄 `data/` 與 `taigi_converter/data/` 同步更新
- 重編根目錄與 package 內 artifacts

### 2026-05-06：銀行服務/交易與計程車雜項補強（round186）
**新增人工詞條（14 條 active；runtime 13 條 + identity 保護 1 條）：**
- `需要抽號碼牌嗎→需要抽號碼牌無`
- `還要等多久→猶愛等偌久`
- `請問開戶櫃檯在哪裡→借問開戶櫃檯佇佗位`
- `營業時間到幾點→營業時間到幾點`
- `手續費可以減免嗎→手續費會當減免無`
- `我要轉帳→我欲轉帳`
- `可以跨行轉帳嗎→會當跨行轉帳無`
- `請幫我刷存摺→請替我刷存摺`
- `匯款需要什麼資料→匯款需要啥物資料`
- `提款有限額嗎→提款有限額無`
- `車牌號碼是多少→車牌號碼是幾號`
- `我東西忘在車上了→我的物件放袂記佇車頂矣`
- `可以聯絡司機嗎→會當聯絡司機無`
- `我在前面下車→我佇頭前落車`

**修正實測錯誤：**
- `車牌號碼是多少？` 不再被轉成金額語境的 `車牌號是偌濟？`
- `請幫我刷存摺。` 收斂為 `請替我刷存摺。`
- `匯款需要什麼資料？` 收斂為 `匯款需要啥物資料？`
- `我東西忘在車上了。` 收斂為較完整的 `我的物件放袂記佇車頂矣。`

**新增迴歸測試：**
- `run_bank_regression.py / bank_service` +5，分類合計 10 筆
- `run_bank_regression.py / bank_transaction` +5，分類合計 10 筆
- `run_taxi_regression.py / misc` +4，分類合計 8 筆
- bank 總數 26→36；taxi 總數 28→32

**資料同步：**
- 根目錄 `data/` 與 `taigi_converter/data/` 同步更新
- 重編根目錄與 package 內 artifacts

### 2026-05-06：飯店入住/訂房與銀行開戶情境補強（round185）
**新增人工詞條（14 條 active）：**
- `我會晚一點到→我會較晏一點到`
- `入住需要證件嗎→入住需要證件無`
- `需要押金嗎→需要押金無`
- `房號是多少→房號是幾號`
- `我想改訂房日期→我想欲改訂房日期`
- `可以加床嗎→會當加床無`
- `訂房有含早餐嗎→訂房有含早頓無`
- `入住人數要改→入住人數愛改`
- `可以取消訂房嗎→會當取消訂房無`
- `開戶需要什麼資料→開戶需要啥物資料`
- `需要帶印章嗎→需要帶印仔無`
- `提款卡什麼時候可以拿→提款卡啥物時陣會當提`
- `可以開通網路銀行嗎→會當開通網路銀行無`
- `資料不齊可以補件嗎→資料無齊會當補件無`

**修正實測錯誤：**
- `房號是多少？` 不再被轉成金額語境的 `房號是偌濟？`
- `入住人數要改。` 收斂為義務語氣 `愛改`
- `我想改訂房日期。` 收斂為 `我想欲改訂房日期。`
- `資料不齊可以補件嗎？` 不再保留華語否定 `不齊`

**新增迴歸測試：**
- `run_hotel_regression.py / check_in` +4，分類合計 8 筆
- `run_hotel_regression.py / reservation` +5，分類合計 10 筆
- `run_bank_regression.py / bank_account` +5，分類合計 10 筆
- hotel 總數 25→34；bank 總數 21→26

**資料同步：**
- 根目錄 `data/` 與 `taigi_converter/data/` 同步更新
- 重編根目錄與 package 內 artifacts

### 2026-05-06：購物瀏覽/比較/付款情境補強（round184）
**新增人工詞條（13 條 active）：**
- `有更便宜的嗎→有閣較俗的無`
- `這個比較耐用嗎→這个較耐用無`
- `有不同大小嗎→敢有大細無仝無`
- `同款有別的顏色嗎→同款有別款顏色無`
- `哪一個比較划算→佗一个較合算`
- `可以用電子支付嗎→會當用電子付錢無`
- `可以分期付款嗎→會當分期付錢無`
- `有會員折扣嗎→有會員拍折無`
- `可以開統一編號嗎→會當開統一編號無`
- `我想找黑色的→我想欲揣烏色的`
- `有這個尺寸嗎→有這个尺寸無`
- `可以推薦一下嗎→會當推薦一下無`
- `店員可以幫我找尺寸嗎→店員會當替我揣尺寸無`

**修正實測錯誤：**
- `有不同大小嗎？` 不再被 `有無→敢有` fluency rule 二次改成 `敢敢有...`
- `店員可以幫我找尺寸嗎？` 不再保留 `幫我找尺寸`
- `可以分期付款嗎？` 收斂為 `會當分期付錢無？`
- `有會員折扣嗎？` 收斂為 `有會員拍折無？`

**新增迴歸測試：**
- `run_shopping_regression.py / browsing` +4，分類合計 8 筆
- `run_shopping_regression.py / payment` +4，分類合計 8 筆
- `run_shopping_regression.py / comparative` +5，分類合計 8 筆
- shopping 總數 29→42

**資料同步：**
- 根目錄 `data/` 與 `taigi_converter/data/` 同步更新
- 重編根目錄與 package 內 artifacts

### 2026-05-06：餐廳服務/口味與購物購買情境補強（round183）
**新增人工詞條（13 條 active）：**
- `可以加水嗎→會當加水無`
- `可以換餐具嗎→會當換餐具無`
- `我的餐還沒來→我的餐猶未來`
- `可以加一張椅子嗎→會當加一張椅仔無`
- `不要香菜→莫芫荽`
- `可以少鹽嗎→會當少鹽無`
- `我對花生過敏→我食塗豆會過敏`
- `我不能吃牛肉→我袂當食牛肉`
- `我想試穿這件→我想欲試穿這件`
- `有大一點的尺寸嗎→有較大的尺寸無`
- `這個還有貨嗎→這个閣有貨無`
- `我要結帳→我欲結數`
- `這件可以刷卡嗎→這件會當刷卡無`

**修正實測錯誤：**
- `可以加一張椅子嗎？` 不再輸出量詞不自然的 `一塊椅仔`
- `我對花生過敏。` 收斂為較自然的 `我食塗豆會過敏。`
- 餐廳服務與口味限制情境補足加水、餐具、催餐、香菜、少鹽與食物禁忌
- 購物購買情境補足試穿、尺寸、庫存、結帳與刷卡

**新增迴歸測試：**
- `run_restaurant_regression.py / service` +4，分類合計 8 筆
- `run_restaurant_regression.py / spice_dietary` +4，分類合計 8 筆
- `run_shopping_regression.py / purchase` +5，分類合計 8 筆
- restaurant 總數 28→36；shopping 總數 24→29

**資料同步：**
- 根目錄 `data/` 與 `taigi_converter/data/` 同步更新
- 重編根目錄與 package 內 artifacts

### 2026-05-06：職場會議/流程/請假情境補強（round182）
**新增人工詞條（12 條 active；runtime 11 條 + identity 保護 1 條）：**
- `會議取消了→會議取消矣`
- `會議延後到下午→會議延後到下晝`
- `請把會議連結寄給我→請共會議連結寄予我`
- `開會前提醒我→開會前共我提醒`
- `請幫我上傳檔案→請替我上傳檔案`
- `我等主管回覆→我等主管回覆`
- `資料要改一下→資料愛改一下`
- `請把新版寄給客戶→請共新版寄予客戶`
- `我今天會晚點到→我今仔日會較晏到`
- `我想請病假→我想欲告病`
- `我臨時有事→我臨時有代誌`
- `我不在座位上→我無佇座位頂`

**修正實測錯誤：**
- `我今天會晚點到。` 不再把 `晚點` 誤轉成 `等陣仔`
- `請幫我上傳檔案。` 不再輸出 `請共我上傳檔案`
- `資料要改一下。` 收斂為義務語氣 `愛改`
- `我不在座位上。` 不再保留華語否定介詞 `不在`

**新增迴歸測試：**
- `run_workplace_regression.py / meeting` +4，分類合計 8 筆
- `run_workplace_regression.py / workflow` +4，分類合計 8 筆
- `run_workplace_regression.py / leave_availability` +4，分類合計 8 筆
- workplace 總數 26→38

**資料同步：**
- 根目錄 `data/` 與 `taigi_converter/data/` 同步更新
- 重編根目錄與 package 內 artifacts

### 2026-05-06：職場位置/進行式與餐廳付款補強（round181）
**新增人工詞條（12 條 active）：**
- `會議室在哪裡→會議室佇佗位`
- `茶水間在走廊旁邊→茶水間佇走廊邊仔`
- `影印機在櫃檯後面→影印機佇櫃檯後壁`
- `我的座位在窗戶旁邊→我的座位佇窗仔門邊仔`
- `我正在整理資料→我佇咧整理資料`
- `他正在確認名單→伊佇咧確認名單`
- `同事正在列印文件→同事佇咧列印文件`
- `我們正在聯絡客戶→咱佇咧聯絡客戶`
- `可以分開結帳嗎→會當分開結數無`
- `我要用現金付→我欲付現錢`
- `可以開發票嗎→會當開發票無`
- `發票可以用載具嗎→發票會當用載具無`

**修正實測錯誤：**
- `影印機在櫃檯後面。` 不再保留華語介詞 `在`
- `我的座位在窗戶旁邊。` 收斂為 `佇窗仔門邊仔`
- `我要用現金付。` 收斂為較自然的 `我欲付現錢。`
- `可以分開結帳嗎？`、`可以開發票嗎？`、`發票可以用載具嗎？` 補進餐廳付款 regression

**新增迴歸測試：**
- `run_workplace_regression.py / office_location` +4，分類合計 7 筆
- `run_workplace_regression.py / progressive` +4，分類合計 7 筆
- `run_restaurant_regression.py / payment` +4，分類合計 8 筆
- workplace 總數 18→26；restaurant 總數 24→28

**資料同步：**
- 根目錄 `data/` 與 `taigi_converter/data/` 同步更新
- 重編根目錄與 package 內 artifacts

### 2026-05-06：家庭親子/手足/祖父母情境補強（round180）
**新增人工詞條（12 條 active）：**
- `孩子該起床了→囡仔愛起床矣`
- `下午去接孩子→下晝去接囡仔`
- `幫孩子換衣服→幫囡仔換衫`
- `不要哭了→毋通哭矣`
- `哥哥妹妹在吵架→阿兄小妹仔佇咧冤家`
- `姐姐幫弟弟拿書包→阿姊幫阿弟仔提冊包`
- `兄弟姐妹要一起分享玩具→兄弟姊妹愛鬥陣分享𨑨迌物仔`
- `妹妹不想寫功課→小妹仔毋想欲寫功課`
- `明天去看爺爺奶奶→明仔載去看阿公阿媽`
- `阿公身體還好嗎→阿公身體敢猶好`
- `奶奶要去散步→阿媽欲去散步`
- `打電話給爺爺→拍電話予阿公`

**修正實測錯誤：**
- `哥哥妹妹在吵架。` 不再保留華語介詞 `在`，收斂為 `佇咧冤家`
- `兄弟姐妹要一起分享玩具。` 不再保留 `兄弟姐妹` 與華語式 `要`
- `妹妹不想寫功課。` 不再輸出 `無想欲`
- `打電話給爺爺。` 不再把 `打電話` 壓成不完整的 `敲`

**新增迴歸測試：**
- `run_family_regression.py / parent_child` +4，分類合計 8 筆
- `run_family_regression.py / siblings` +4，分類合計 8 筆
- `run_family_regression.py / grandparents` +4，分類合計 8 筆
- family 總數 28→40

**資料同步：**
- 根目錄 `data/` 與 `taigi_converter/data/` 同步更新
- 重編根目錄與 package 內 artifacts

### 2026-05-06：家庭日常/照護情境補強（round179）
**新增人工詞條（8 條 active）：**
- `孩子要吃藥→囡仔愛食藥仔`
- `藥要按時吃→藥仔愛照時間食`
- `媽媽要休息→阿母欲歇睏`
- `我陪你去看醫生→我陪你去予醫生看`
- `爸爸要出門了→阿爸欲出門矣`
- `晚餐好了→暗頓好矣`
- `我們一起吃飯→咱鬥陣食飯`
- `記得洗碗→記著洗碗`

**修正實測錯誤：**
- `孩子要吃藥。` 不再保留華語式 `要`，收斂為照護語境的 `愛`
- `藥要按時吃。` 不再保留 `按時` 與單字 `藥`，收斂為 `藥仔愛照時間食`
- `晚餐好了。` 補上 `暗頓`，避免保留華語 `晚餐`
- `爸爸要出門了。`、`媽媽要休息。` 收斂為意願/即將動作的 `欲`

**新增迴歸測試：**
- `run_family_regression.py / health_care` +4，分類合計 8 筆
- `run_family_regression.py / daily` +4，分類合計 8 筆
- family 總數 20→28

**資料同步：**
- 根目錄 `data/` 與 `taigi_converter/data/` 同步更新
- 重編根目錄與 package 內 artifacts

### 2026-05-06：學校作業/校園動線補強（round178）
**新增人工詞條（2 條 active）：**
- `作業明天要交→作業明仔載愛交`
- `我要去圖書館還書。→我欲去圖冊館還書。`

**修正實測錯誤：**
- `作業明天要交。` 收斂義務語氣為 `愛交`
- `我要去圖書館還書。` 用句級 override 避開 `還` 被二次改成 `猶` 的問題，並保留既有 round137 的 `還書` identity 政策

**新增迴歸測試：**
- `run_school_regression.py / homework` +3，分類合計 6 筆
- `run_school_regression.py / campus` +4，分類合計 8 筆
- school 總數 26→33

**資料同步：**
- 根目錄 `data/` 與 `taigi_converter/data/` 同步更新
- 重編根目錄與 package 內 artifacts

### 2026-05-06：學校能力/考試情境補強（round177）
**新增人工詞條（2 條 active）：**
- `不會寫這題→袂曉寫這題`
- `考試日期改到→考試日期改做`

**修正實測錯誤：**
- `我不會寫這題。` 不再輸出偏「未來不寫」的 `我袂寫這題。`
- `考試日期改到下星期。` 和既有預約/時間改期風格一致，收斂為 `改做下禮拜`

**新增迴歸測試：**
- `run_school_regression.py / ability` +3，分類合計 5 筆
- `run_school_regression.py / exam` +3，分類合計 6 筆
- school 總數 20→26

**資料同步：**
- 根目錄 `data/` 與 `taigi_converter/data/` 同步更新
- 重編根目錄與 package 內 artifacts

### 2026-05-06：計程車付款/臨停補強（round176）
**新增人工詞條（2 條 active）：**
- `靠邊停一下→靠路邊停一下`
- `臨停一下→暫停一下`

**修正實測錯誤：**
- `麻煩靠邊停一下。` 收斂為較自然的 `麻煩靠路邊停一下。`
- `前面臨停一下就好。` 不再保留偏書面/交通管制語境的 `臨停`

**新增迴歸測試：**
- `run_taxi_regression.py / navigation` +2，分類合計 7 筆
- `run_taxi_regression.py / payment` +3，分類合計 7 筆
- taxi 總數 23→28

**資料同步：**
- 根目錄 `data/` 與 `taigi_converter/data/` 同步更新
- 重編根目錄與 package 內 artifacts

### 2026-05-06：計程車叫車/目的地補強（round175）
**新增人工詞條（2 條 active）：**
- `幫我叫一台計程車→替我叫一台計程車`
- `請送我到機場→請載我到機場`

**修正實測錯誤：**
- `可以幫我叫一台計程車嗎？` 不再保留華語 `幫我`
- `請送我到機場。` 收斂為計程車語境較自然的 `請載我到機場。`

**新增迴歸測試：**
- `run_taxi_regression.py / hailing` +2，分類合計 5 筆
- `run_taxi_regression.py / destination` +2，分類合計 5 筆
- taxi 總數 19→23

**資料同步：**
- 根目錄 `data/` 與 `taigi_converter/data/` 同步更新
- 重編根目錄與 package 內 artifacts

### 2026-05-06：飯店設施與問題反映補強（round174）
**新增人工詞條（3 條 active，runtime 2 條 + identity 保護 1 條）：**
- `吹風機→吹風機`
- `洗衣服務→洗衫服務`
- `房卡打不開房門→房卡開袂開房門`

**修正實測錯誤：**
- `請問有吹風機嗎？` 不再輸出像電扇的 `搧風機`
- `請問有洗衣服務嗎？` 不再被拆成 `洗衫務`
- `房卡打不開房門。` 不再輸出不自然的 `拍袂開`

**新增迴歸測試：**
- `run_hotel_regression.py / amenities` +2，分類合計 5 筆
- `run_hotel_regression.py / issues` +2，分類合計 5 筆
- hotel 總數 21→25

**資料同步：**
- 根目錄 `data/` 與 `taigi_converter/data/` 同步更新
- 重編根目錄與 package 內 artifacts

### 2026-05-06：飯店退房/寄放行李補強（round173）
**新增人工詞條（2 條 active）：**
- `延後退房→較晏退房`
- `房卡要交回櫃檯→房卡愛交轉去櫃檯`

**修正實測錯誤：**
- `可以延後退房嗎？` 不再被舊詞條拆成錯誤的 `延倒勼房`
- `房卡要交回櫃檯嗎？` 收斂為 `房卡愛交轉去櫃檯無？`

**新增迴歸測試：**
- `run_hotel_regression.py / check_out` +4，分類合計 6 筆
- hotel 總數 17→21

**資料同步：**
- 根目錄 `data/` 與 `taigi_converter/data/` 同步更新
- 重編根目錄與 package 內 artifacts

### 2026-05-06：醫療住院/病房流程補強（round172）
**新增人工詞條（3 條 active）：**
- `陪病家屬要先登記→陪病家屬愛先登記`
- `到護理站報到→去護理站報到`
- `轉到普通病房→轉去普通病房`

**修正實測錯誤：**
- `陪病家屬要先登記。` 不再整句保留華語
- `請先到護理站報到。` 收斂為較口語的 `請先去護理站報到。`
- `這位病人明天要轉到普通病房。` 收斂為 `轉去普通病房`

**新增迴歸測試：**
- `run_medical_regression.py / rooms_inpatient` +3，分類合計 11 筆
- medical 總數 69→72

**資料同步：**
- 根目錄 `data/` 與 `taigi_converter/data/` 同步更新
- 重編根目錄與 package 內 artifacts

### 2026-05-06：購物售後取消訂單/退款補強（round171）
**新增人工詞條（3 條 active）：**
- `需要取消訂單→欲取消訂單`
- `想取消訂單→想欲取消訂單`
- `退款→退錢`

**修正實測錯誤：**
- `我需要取消訂單。` 不再整句保留華語
- `我想取消訂單。` 收斂為 `我想欲取消訂單。`
- `可以退款嗎？` 不再保留書面 `退款`

**新增迴歸測試：**
- `run_shopping_regression.py / after_sales` +4，分類合計 6 筆
- shopping 總數 20→24

**資料同步：**
- 根目錄 `data/` 與 `taigi_converter/data/` 同步更新
- 重編根目錄與 package 內 artifacts

### 2026-05-06：餐廳座位/候位情境補強（round170）
**新增人工詞條（3 條 active）：**
- `要等多久→愛等偌久`
- `兒童椅→囡仔椅`
- `靠窗的位置→靠窗的位`

**修正實測錯誤：**
- `請問要等多久？` 不再保留義務/需求語境不自然的 `要等`
- `可以幫我安排兒童椅嗎？` 不再保留華語 `兒童椅`
- `可以坐靠窗的位置嗎？` 不再保留華語 `位置`

**新增迴歸測試：**
- `run_restaurant_regression.py / seating` +4，分類合計 6 筆
- restaurant 總數 20→24

**資料同步：**
- 根目錄 `data/` 與 `taigi_converter/data/` 同步更新
- 重編根目錄與 package 內 artifacts

### 2026-05-06：交通 crowd/safety 薄弱類別補強（round169）
**新增人工詞條（2 條 active）：**
- `人很多的時候→人濟的時陣`
- `月台邊緣→月台邊仔`

**修正實測錯誤：**
- `人很多的時候，請排隊不要插隊。` 不再被拆成錯誤的 `人規千萬時陣`
- `請不要靠近月台邊緣。` 收斂為較自然的 `請莫靠近月台邊仔。`

**新增迴歸測試：**
- `run_transport_regression.py / crowd_safety` +2，分類合計 10 筆
- transport 總數 60→62

**資料同步：**
- 根目錄 `data/` 與 `taigi_converter/data/` 同步更新
- 重編根目錄與 package 內 artifacts

### 2026-04-29：日常回應「願意/肯傾聽」口語自然度補強（round155）
**新增人工詞條（4 條 active）：**
- `我都願意傾聽喔→我攏肯用心聽你講喔`
- `我都肯傾聽喔→我攏肯聽你講喔`
- `我都願意傾聽→我攏肯用心聽你講`
- `我都肯傾聽→我攏肯聽你講`

**修正實測錯誤：**
- `我都願意傾聽喔` 不再輸出偏華語的 `我都肯傾聽喔`
- `我都肯傾聽喔` 不再保留書面 `傾聽`
- 依使用者校正，`願意傾聽` 採較溫柔正式的 `肯用心聽你講`，`肯傾聽` 採較口語的 `肯聽你講`

**新增迴歸測試：**
- `run_conversation_regression.py / daily_response` +4，分類合計 11 筆
- 覆蓋有無句尾 `喔` 的兩組傾聽回應

**資料同步：**
- 根目錄 `data/` 與 `taigi_converter/data/` 同步更新
- 重編根目錄與 package 內 artifacts

### 2026-04-29：公車等車時間問句語序收斂（round154）
**新增人工詞條（6 條 active）：**
- `還要再等多久→猶愛閣等偌久`
- `還要再等幾分鐘→猶愛閣等幾分鐘`
- `大概要再等多久→差不多閣愛等偌久`
- `大概要再等幾分鐘→差不多閣愛等幾分鐘`
- `要再等多久→愛閣等偌久`
- `要再等幾分鐘→愛閣等幾分鐘`

**修正實測錯誤：**
- `還要再等多久？` 不再輸出 `閣愛再等偌久？`，改為 `猶愛閣等偌久？`
- `大概要再等多久？` 不再輸出 `差不多要再等偌久？`，改為 `差不多閣愛等偌久？`
- `要再等多久才有車？` 輸出 `愛閣等偌久才有車？`

**新增迴歸測試：**
- `run_bus_regression.py / bus_time_queries` 再 +6，分類合計 53 筆
- 覆蓋還要再等多久/幾分鐘、大概要再等多久/幾分鐘、要再等多久/幾分鐘才有車

**資料同步：**
- 根目錄 `data/` 與 `taigi_converter/data/` 同步更新
- 重編根目錄與 package 內 artifacts

### 2026-04-29：公車首末班時間問句一致化（round153）
**新增人工詞條（1 條 active）：**
- `末班→尾班`

**修正實測錯誤：**
- `末班時間是幾點？` 不再保留華語 `末班`，輸出 `尾班時間是幾點？`
- `末班發車時間是幾點？` 輸出 `尾班開車時間是幾點？`
- `末班大概什麼時候開？` 輸出 `尾班差不多啥物時陣開？`

**新增迴歸測試：**
- `run_bus_regression.py / bus_time_queries` 再 +8，分類合計 47 筆
- 覆蓋發車時間、到站時間、首班/末班時間、首班/末班發車時間、下一班到站時間與末班大概何時開

**資料同步：**
- 根目錄 `data/` 與 `taigi_converter/data/` 同步更新
- 重編根目錄與 package 內 artifacts

### 2026-04-29：公車營運/行駛時間問句收斂（round152）
**新增人工詞條（6 條 active）：**
- `有營運嗎→有開無`
- `正常營運嗎→正常開無`
- `有行駛嗎→有開無`
- `照常行駛嗎→照常開無`
- `正常行駛嗎→正常開無`
- `停駛還是正常營運→停開猶是正常開`

**修正實測錯誤：**
- `今天有營運嗎？` 不再保留華語 `營運`，輸出 `今仔日有開無？`
- `假日有行駛嗎？` 不再保留華語 `行駛`，輸出 `假日有開無？`
- `這班車今天照常行駛嗎？` 輸出 `這班車今仔日照常開無？`
- `今天停駛還是正常營運？` 輸出 `今仔日停開猶是正常開？`

**新增迴歸測試：**
- `run_bus_regression.py / bus_time_queries` 再 +9，分類合計 39 筆
- 覆蓋今天/明天/假日/國定假日/颱風天/春節期間的公車營運與行駛時間狀態問句

**資料同步：**
- 根目錄 `data/` 與 `taigi_converter/data/` 同步更新
- 重編根目錄與 package 內 artifacts

### 2026-04-29：公車班距與營運時間查詢覆蓋（round151）
**新增人工詞條（5 條 active）：**
- `每小時一班→每一點鐘一班`
- `每多久一班→偌久一班`
- `每幾分鐘一班→偌久一班`
- `每隔多久一班→隔偌久一班`
- `每隔幾分鐘一班→隔偌久一班`

**修正實測錯誤：**
- `每小時一班。` 不再輸出 `每點鐘一班。`，改為 `每一點鐘一班。`
- `尖峰時間每幾分鐘一班？` 不再輸出 `尖峰時間每偌久一班？`，改為 `尖峰時間偌久一班？`
- `離峰時間每隔幾分鐘一班？` 不再輸出 `離峰時間每隔偌久一班？`，改為 `離峰時間隔偌久一班？`

**新增迴歸測試：**
- `run_bus_regression.py / bus_time_queries` 再 +10，分類合計 30 筆
- 覆蓋班距大概多久、每小時一班、平日/假日半小時班距、每多久/每幾分鐘/每隔幾分鐘一班、收班時間、路線營運時間、服務時間與預計到站時間

**資料同步：**
- 根目錄 `data/` 與 `taigi_converter/data/` 同步更新
- 重編根目錄與 package 內 artifacts

### 2026-04-29：公車首末班、班距與時刻表時間查詢覆蓋（round150）
**新增人工詞條（2 條 active）：**
- `首班車→頭班車`
- `首班→頭班`

**修正實測錯誤：**
- `首班車幾點發車？` 不再保留華語 `首班車`，輸出 `頭班車幾點開車？`
- 和既有 `頭班車明天會晚半小時。` / `末班車→尾班車` 的公車語域保持一致

**新增迴歸測試：**
- `run_bus_regression.py / bus_time_queries` 再 +10，分類合計 20 筆
- 覆蓋首班車/末班車、平日/假日班距、尖峰/離峰班距、延誤多久、時刻表生效日、改點後第一班、臨時班車到站時間

**資料同步：**
- 根目錄 `data/` 與 `taigi_converter/data/` 同步更新
- 重編根目錄與 package 內 artifacts

### 2026-04-29：公車查詢時間問句與月初/月尾補強（round149）
**新增人工詞條（15 條 active）：**
- 公車查詢會用到的星期/週問句：`星期幾→禮拜幾`、`週幾→禮拜幾`、`周幾→禮拜幾`
- 查班日期問句：`哪一天→佗一工`、`哪天→佗一工`
- 行車/恢復/等待時長問句：`多長時間→偌久`、`多少時間→偌久`、`需要多長時間→愛偌久`、`需要多少時間→愛偌久`、`要花多長時間→愛偌久`、`要花多少時間→愛偌久`、`還要多長時間→閣愛偌久`
- 時刻表月份時間：`下個月→後個月`、`月初→月頭`、`月底→月尾`

**修正實測錯誤：**
- `末班車星期幾比較早？` 不再保留 `星期幾`，輸出 `尾班車禮拜幾較早？`
- `哪一天有加班車？` 不再輸出 `佗一天`，改為 `佗一工`
- `從斗六到北港需要多少時間？` 不再輸出 `偌濟時間`
- `這條路線還要多長時間才會恢復？` 不再輸出 `偌長時間`
- `下個月月初時刻表會改嗎？` 可輸出 `後個月月頭時刻表會改無？`
- `這班車月底會停駛嗎？` 可輸出 `這班車月尾會停開無？`

**新增迴歸測試：**
- `run_bus_regression.py / bus_time_queries` +10
- 覆蓋下一班到站、班距、星期/週、哪天、時長、下個月、月初/月尾等公車查詢時間句

**資料同步：**
- 根目錄 `data/` 與 `taigi_converter/data/` 同步更新
- 重編根目錄與 package 內 artifacts

### 2026-04-29：台灣好行雲林四線優惠店家與休憩點補強（round148）
**新增人工詞條（24 條 active）：**
- 斗六古坑線優惠/休憩點：`蜜蜂故事館`、`鄉村休閒農莊`、`咖啡大街民宿`、`莿桐花咖啡`、`石墩庭園咖啡`、`華山咖啡大街物產館`
- 雲西線/北港虎尾線休憩點：`蝦の故鄉興義軒休閒園區`、`蝦の故鄉 興義軒休閒園區→蝦の故鄉興義軒休閒園區`、`北港當歸鴨`、`阿甘薯叔雲林高鐵門市`、`阿甘薯叔 雲林高鐵門市→阿甘薯叔雲林高鐵門市`
- Locative/固定片語：`在蜜蜂故事館→佇蜜蜂故事館`、`在鄉村休閒農莊→佇鄉村休閒農莊`、`在咖啡大街民宿→佇咖啡大街民宿`、`在莿桐花咖啡→佇莿桐花咖啡`、`在石墩庭園咖啡→佇石墩庭園咖啡`、`在華山咖啡大街物產館→佇華山咖啡大街物產館`、`在華山咖啡大街→佇華山咖啡大街`、`在蝦の故鄉興義軒休閒園區→佇蝦の故鄉興義軒休閒園區`、`在北港當歸鴨→佇北港當歸鴨`、`在阿甘薯叔雲林高鐵門市→佇阿甘薯叔雲林高鐵門市`、`都在華山咖啡大街→攏佇華山咖啡大街`

**Rule 擴充：**
- 新增 `rl_148_yunlin_shuttle_food_zai_locative_places`，支援台灣好行雲林四線優惠店家與休憩點作地點時 `在X→佇X`

**修正實測錯誤：**
- `站牌在蜜蜂故事館前面嗎？` 不再保留華語介詞 `在`
- `石墩庭園咖啡在華山咖啡大街附近嗎？` 可輸出 `石墩庭園咖啡佇華山咖啡大街附近無？`
- 官方店名含空白的 `蝦の故鄉 興義軒休閒園區`、`阿甘薯叔 雲林高鐵門市` 會收斂成穩定專名

**新增迴歸測試：**
- `run_bus_regression.py / yunlin_shuttle_food_stops` +7
- 覆蓋斗六古坑線優惠店家、雲西線休憩點、站牌 locative、官方店名空白正規化

**資料同步：**
- 根目錄 `data/` 與 `taigi_converter/data/` 同步更新
- 重編根目錄與 package 內 artifacts

### 2026-04-28：斗六古坑線與西螺周邊景點補強（round147）
**新增人工詞條（19 條 active）：**
- 北港虎尾線/高鐵雲林站周邊：`聖玫瑰天主堂`、`西螺廣福宮`、`西螺媽廟`、`西螺大橋`、`斗六棒球場→斗六野球場`
- 斗六古坑線雲中街周邊與優惠店家：`凹凸咖啡`、`猿樂作`、`貝歐克尼Balcony乾燥花→貝歐克尼Balcony焦燥花`、`黑膠音樂故事館`、`Mr. Lobby Coffee Roaster`、`劍湖山世界樂園`
- Locative/固定片語：`在聖玫瑰天主堂→佇聖玫瑰天主堂`、`在西螺廣福宮→佇西螺廣福宮`、`在西螺大橋→佇西螺大橋`、`在斗六棒球場→佇斗六野球場`、`在雲中街→佇雲中街`、`在黑膠音樂故事館→佇黑膠音樂故事館`、`都在高鐵雲林站北邊→攏佇高鐵雲林站北爿`、`都在雲中街→攏佇雲中街`

**Rule 擴充：**
- 新增 `rl_147_yunlin_douliu_xiluo_zai_locative_places`，支援斗六古坑線與北港虎尾線周邊景點/店家作地點時 `在X→佇X`

**修正實測錯誤：**
- 依使用者校正，`斗六棒球場` 台語輸出維持 `斗六野球場`
- 依使用者校正，地名/店名後的普通名詞要翻譯；`貝歐克尼Balcony乾燥花` 維持輸出 `貝歐克尼Balcony焦燥花`
- `都在高鐵雲林站北邊` 不再輸出 `攏咧高鐵雲林站北爿`

**新增迴歸測試：**
- `run_bus_regression.py / yunlin_douliu_xiluo_attractions` +7
- 覆蓋聖玫瑰天主堂、西螺廣福宮/大橋、斗六棒球場、雲中街店家與台灣好行優惠問句

**資料同步：**
- 根目錄 `data/` 與 `taigi_converter/data/` 同步更新
- 重編根目錄與 package 內 artifacts

### 2026-04-28：雲林草嶺線小站名補強（round146）
**新增人工詞條（20 條 active）：**
- 草嶺線小站名：`受天宮`、`東和`、`早寮`、`二坪仔`、`東內寮`、`小旗仔`、`檳榔宅`、`外湖`、`內湖`、`草嶺`
- 對應 locative 片語：`在受天宮→佇受天宮`、`在東和→佇東和`、`在早寮→佇早寮`、`在二坪仔→佇二坪仔`、`在東內寮→佇東內寮`、`在小旗仔→佇小旗仔`、`在檳榔宅→佇檳榔宅`、`在外湖→佇外湖`、`在內湖→佇內湖`、`在草嶺→佇草嶺`

**Rule 擴充：**
- 新增 `rl_146_yunlin_caoling_minor_zai_locative_places`，支援草嶺線小站名作地點時 `在X→佇X`

**修正實測錯誤：**
- `檳榔宅` 不再被一般食物詞彙拆成 `菁仔宅`
- `站牌在檳榔宅附近嗎？` 可輸出 `站牌佇檳榔宅附近無？`

**新增迴歸測試：**
- `run_bus_regression.py / yunlin_caoling_minor_stops` +6
- 覆蓋草嶺線小站名、`檳榔宅` 專名保護、站牌 locative 與 `外湖/內湖` 動線問句

**資料同步：**
- 根目錄 `data/` 與 `taigi_converter/data/` 同步更新
- 重編根目錄與 package 內 artifacts

### 2026-04-28：雲林斗南、沿海與草嶺石壁景點補強（round145）
**新增/調整人工詞條（21 條 active，1 條 disabled）：**
- 斗南：`斗南他里霧文化園區`、`他里霧文化園區`、`斗南圓環`
- 北港美食：`北港限定早餐`、`北港巷弄美食`、`北港美食小吃`
- 雲西沿海：`金湖沙灘`、`四湖參天宮`、`三條崙海水浴場`、`箔子寮漁港`
- 草嶺/石壁：`草嶺風景區`、`石壁風景區`、`草嶺古道`、`大飛山`
- 虎尾：`虎尾市區`、`虎尾建國一村`
- 補正規化後專名 `斗南他裡霧文化園區→斗南他里霧文化園區`、`他裡霧文化園區→他里霧文化園區`、`他裡霧→他里霧`
- 啟用 `他`、`里` 右側 `里霧`/`霧` 語境保護；停用 `裡→里` char 草案，避免影響 `厝裡`

**Rule 擴充：**
- 新增 `rl_145_yunlin_coastal_caoling_zai_locative_places`，支援斗南、雲西沿海、草嶺/石壁景點作地點時 `在X→佇X`

**Runtime 修正：**
- 修正 context-aware char entries 仍進入一般 `char_map` 的問題；現在 context char 只走 contextual pass，避免 `他在點菜` 被專名保護誤改成 `他佇咧叫菜`

**修正實測錯誤：**
- `斗南他里霧文化園區` 不再輸出 `斗南伊裡霧文化園區` 或 `斗南伊里霧文化園區`
- `北港限定早餐` 不再被拆成 `北港限定早頓`
- `北港美食小吃` 不再被拆成 `北港美食小食`
- `金湖沙灘` 不再被拆成 `金湖海坪`

**新增迴歸測試：**
- `run_bus_regression.py / yunlin_coastal_caoling` +9
- 覆蓋斗南他里霧、北港美食、雲西沿海、草嶺石壁與虎尾市區站牌問句

**資料同步：**
- 根目錄 `data/` 與 `taigi_converter/data/` 同步更新
- 重編根目錄與 package 內 artifacts

### 2026-04-28：雲林縣政府推薦遊程景點補強（round144）
**新增人工詞條（28 條）：**
- 土庫/虎尾：`土庫順天宮`、`土庫庄役場`、`土庫老街`、`源順芝麻觀光油廠`
- 斗六：`圖南咖啡故事館`、`行啟記念館`、`三小市集`、`朝露魚鋪觀光工廠`、`石榴車站`、`榴中社區`、`新德豐碾米廠→新德豐米絞`、`張氏宗祠`
- 古坑：`蘿莎玫瑰莊園`、`劍湖山世界幸福摩天輪`、`華山文學步道`、`幽情谷步道`、`水濂洞`、`峭壁雄風步道`、`雲嶺之丘`、`五元二角`
- 北港/口湖：`北港義民廟`、`武德宮`、`沐藝堂`
- 補 `都在土庫→攏佇土庫`

**Rule 擴充：**
- 新增 `rl_144_yunlin_tourism_zai_locative_places`，支援雲林縣政府推薦遊程景點作地點時 `在X→佇X`

**修正實測錯誤：**
- `土庫順天宮、土庫庄役場...都在土庫嗎？` 不再輸出 `攏咧土庫`
- 依使用者校正，`新德豐碾米廠` 應輸出 `新德豐米絞`

**新增迴歸測試：**
- `run_bus_regression.py / yunlin_tourism_attractions` +9
- 覆蓋土庫、斗六、古坑、北港、口湖推薦遊程景點專名

**資料同步：**
- 根目錄 `data/` 與 `taigi_converter/data/` 同步更新
- 重編根目錄與 package 內 artifacts

### 2026-04-28：台灣好行雲林四線站名補強（round143）
**新增人工詞條（39 條）：**
- 斗六古坑線站名：`雲中街`、`社口遊客中心`、`雲林科技大學`、`古坑嘉興宮`、`福祿壽酒廠`、`永光故事屋`、`劍湖山世界`、`華山咖啡大街`
- 雲西線站名：`北港武德宮`、`北港春生活博物館`、`黃金蝙蝠生態館`、`戰水鯨湖廣場`、`顏厝寮聚落`、`北港1911好庫文化產業園區`、`北港遊客中心`、`高鐵嘉義站`
- 雲林草嶺線站名：`鎮西國小`、`水岸藝術公園`、`成大醫院`、`荷苞山桐花公園`、`新草嶺國小站牌`、`草嶺公園`、`東𤧥山莊`
- 補 `都在雲西線→攏佇雲西線`、`都在草嶺線→攏佇草嶺線`、`都在斗六古坑線→攏佇斗六古坑線`

**Rule 擴充：**
- 新增 `rl_143_yunlin_shuttle_zai_locative_places`，支援台灣好行雲林其他路線站名作地點時 `在X→佇X`

**修正實測錯誤：**
- `黃金蝙蝠生態館` 不再被拆成 `黃金夜婆生態館`
- `北港1911好庫文化產業園區` 不再被數字正規化成中文數字
- `成大醫院` 作站名時保留專名，不改成 `成大病院`
- `新草嶺國小站牌` 不再輸出 `國細站牌`

**新增迴歸測試：**
- `run_bus_regression.py / yunlin_shuttle_routes` +10
- 覆蓋斗六古坑線、雲西線、雲林草嶺線站名與 locative 句型

**資料同步：**
- 根目錄 `data/` 與 `taigi_converter/data/` 同步更新
- 重編根目錄與 package 內 artifacts

### 2026-04-28：雲林公車站名與景點專名保護（round142）
**新增人工詞條（60 條 active）：**
- 台灣好行雲林四線常見站名/景點：`北港朝天宮`、`口湖遊客中心`、`三條崙海清宮`、`麥寮拱範宮`、`千巧谷牛樂園牧場`
- 雲林旅遊景點專名：`西螺延平老街`、`丸莊醬油觀光工廠`、`虎尾驛`、`虎尾糖廠`、`雲林故事館`、`古坑綠色隧道`、`水道頭文化園區`、`成龍濕地` 等
- 雲林地名 locative 片語：`在虎尾→佇虎尾`、`在北港→佇北港`、`在高鐵雲林站→佇高鐵雲林站` 等
- `高` + `鐵` 右側語境保護，避免 `高鐵` 在 locative 片語輸出後被 char layer 改成 `懸鐵`

**Rule 擴充：**
- 新增 `rl_142_yunlin_zai_locative_places`，支援雲林站名/景點作地點時 `在X→佇X`

**停用過度泛化草案（1 條）：**
- `站牌在→站牌佇` disabled；避免搶先吃掉 `在高鐵雲林站`，造成 `高鐵` 專名保護失效

**新增迴歸測試：**
- `run_bus_regression.py / yunlin_stops_attractions` +9
- 覆蓋 `北港朝天宮站牌在水道頭文化園區旁邊嗎？→北港朝天宮站牌佇水道頭文化園區隔壁無？`
- 覆蓋 `西螺福興宮和丸莊醬油觀光工廠附近有站牌嗎？→西螺福興宮和丸莊醬油觀光工廠附近有站牌無？`
- 覆蓋 `站牌在高鐵雲林站旁邊嗎？→站牌佇高鐵雲林站隔壁無？`

**資料同步：**
- 根目錄 `data/` 與 `taigi_converter/data/` 同步更新
- 重編根目錄與 package 內 artifacts

### 2026-04-26：時間長度 `一個多月` 修正（round141）
**停用錯誤機器詞條（1 條）：**
- `一個多月→個捅月`（machine review_queue，輸出缺字且語義不通）

**新增人工詞條（1 條）：**
- `一個多月→一个外月`

**新增迴歸測試：**
- `run_conversation_regression.py / schedule_plans` +1
- 覆蓋 `我等了一個多月。→我等一个外月。`

**資料同步：**
- 根目錄 `data/` 與 `taigi_converter/data/` 同步更新
- 重編根目錄與 package 內 artifacts

### 2026-04-26：意圖問句 `打算去...` 修正 + package data 同步（round140）
**停用錯誤機器詞條（1 條）：**
- `打算→拍算`（machine review_queue，會污染 `你打算去哪...` 意圖句）

**新增人工詞條（3 條）：**
- `打算去→欲去`
- `哪邊→佗位`
- `哪座→佗一座`

**新增迴歸測試：**
- `run_conversation_regression.py / schedule_plans` +2
- 覆蓋 `你打算去哪邊游泳呢？→你欲去佗位泅水咧？`
- 覆蓋 `你打算去哪座山呢？→你欲去佗一座山咧？`

**資料同步：**
- 根目錄 `data/` 與 `taigi_converter/data/` 同步更新
- 重編根目錄與 package 內 artifacts，避免 CLI 與安裝套件行為不一致

### 2026-04-25：辦公/會議情境開拓（round139）
**新增詞條（6 條）：**
- `要開會→欲開會`
- `想請→想欲請`
- `再確認→閣確認`
- `再討論→閣討論`
- `會議改到→會議改做`
- `要簽名→愛簽名`

**Rule 擴充：**
- `rl_g_zai_locative_places` 加入 `會議室`，修正 `主管在會議室→主管佇會議室`
- 新增 `rl_2ff8cda43199`：辦公名詞主語進行式，支援 `同事/主管/老闆/助理/客戶/工程師/設計師在V → X佇咧V`

**新增迴歸測試：**
- `run_workplace_regression.py`（18 筆，5 類：meeting、office_location、progressive、workflow、leave_availability）

**資料同步：**
- 根目錄 `data/` 與 `taigi_converter/data/` 同步更新並重編 artifacts

**驗證：**
- 全部 regression 通過（422 筆）

### 2026-04-25：還給+代詞保護 + package data 同步（round138）
**新增詞條（9 條）：**
- `還給我`、`還給你`、`還給他`、`還給她`、`還給您`
- `還給我們`、`還給咱`、`還給你們`、`還給他們`
- 修正 round137 已知限制：`還給你` 不再被 `給你→予你` overlap 繞過，避免輸出 `猶予你`

**資料同步：**
- 將根目錄 `data/` 主資料同步到 `taigi_converter/data/`
- 重編根目錄與 package 內 artifacts，避免 CLI 與安裝套件行為不一致

**迴歸更新：**
- `run_conversation_regression.py / daily_response` +1：`還給你。→還給你。`
- 全部 regression 通過（404 筆）

### 2026-04-23：基本文法大補強（round137）
**新增詞條（~100 條）：**

**最→上（最高級）：**
- `最好→上好`、`最多→上多`、`最快→上緊`、`最大→上大`、`最高→上懸` 等 21 條
- `最好的→上好的`、`最重要的→上要緊的` 等 13 條含「的」形式（修正 base seed `最好的→一粒一` 干擾）

**更→閣較（比較級）：**
- `更好→閣較好`、`更大→閣較大`、`更快→閣較緊`、`更便宜→閣較俗`、`更好吃→閣較好食` 等 25 條

**想欲（want to 結構）：**
- `想去→想欲去`、`想吃→想欲食`、`想睡→想欲睏`、`想喝→想欲啉`、`想回去→想欲轉去` 等 16 條

**讓+代詞 → 予：**
- `讓他→予伊`、`讓她→予伊`、`讓他們→予怹`（補 Mandarin 原形詞條）
- `讓伊→予伊`、`讓怹→予怹`（補 post-conversion 形）

**還（動詞歸還）保護：**
- `歸還`、`還款`、`還錢`、`還清`、`還債`、`償還`、`還車`、`還書`、`還給` identity 保護
- 修正 `還→猶` 單字詞條誤轉 return-verb 的問題
- **已知限制**：`還給你` 因 `給你→予你`（3字）overlap，identity mask 被繞過，仍輸出 `猶予你`

**Regression 更新：**
- `run_medical_regression.py` registration 第 30 筆：`我想問` → `我想欲問`
- `run_transport_regression.py` destinations 第 59 筆：`最遠` → `上遠`

### 2026-04-13：X一點比較詞組 + 說→講篇章語 + 雜項修正（round135）
**新增詞條（19 條）：**

**X一點比較詞組（12 條）：**
- `大一點的→較大的`、`小一點的→較細的`（shopping，priority 1300）
- `大一點→較大`、`小一點→較細`（bare comparative，priority 1200）
- `多一點→加一寡`、`少一點→少一寡`（quantity）
- `慢一點→慢一咧`、`早一點→早一咧`（request）
- `高一點的→較懸的`、`低一點的→較低的`
- `長一點的→較長的`、`短一點的→較短的`

**說→講篇章語（4 條）：**
- `說起來→講起來`、`說到底→講到底`
- `換句話說→換句話講`、`也就是說→也就是講`

**其他（3 條）：**
- `那樣子→按呢`（修正 `那樣` + 殘字 `子` 問題，priority 1100）
- `雖然如此→雖然按呢`
- `說老實話→講實在話`（priority 1300，蓋過 `老實→古意` 干擾）

**迴歸更新：**
- `run_medical_regression.py` pharmacy_payment 第 55 筆：`早一點` 預期值更新為 `早一咧`

### 2026-04-12：好X 強化詞 + 在+地點規則擴充 + base seed 清查（round134）
**停用惡性 base seed（5 條）：**
- `算了→煞煞去`（wrong，改為 `算矣`）
- `感冒了→感著`（shadowed `感冒→寒著`，已停用）
- `好吃得很→箭竹仔筍`（完全錯誤）
- `好攻擊→臭腥仔`（完全錯誤）
- `好痛喔→膨疱`（完全錯誤）

**新增詞條（8 條）：**
- `算了→算矣`
- `好餓→真枵`、`好累→真累`、`好熱→真熱`、`好冷→真寒`
- `好痛→真痛`、`好無聊→真無聊`、`好開心→真歡喜`
- 補足了 `很X→真X` 的 `好X→真X` 缺口

**Rule 擴充（rl_g_zai_locative_places）：**
- 加入常見生活地點：公園、餐廳、客廳、廚房、醫院/病院、教室、圖書館/圖冊館、房間、超市、飯店、旅館、銀行、郵局、機場、車站、停車場、操場、體育館、游泳池、浴室、陽台、走廊、電梯、廁所/便所、書房、臥室、醫生館
- 現在 `在公園/餐廳/廚房...` 都能正確轉為 `佇X`

**迴歸測試更新（2 筆）：**
- `run_bus_regression.py`、`run_transport_regression.py` 中 `在病院門跤口` → `佇病院門跤口`

### 2026-04-11：跨領域 modal 誤譯批次修正（round133）
**新增詞條（8 條）+ 修正既有句級 1 條：**
- `如果你要取消掛號→若是你欲取消掛號`
- `如果你要看報告→若是你欲看報告`
- `如果你要申請病歷→若是你欲申請病歷`
- `如果你要找失物→若是你欲找失物`
- `如果你要補票→若是你欲補票`
- `如果你要轉火車→若是你欲轉火車`
- `如果你要查公車到哪裡→若是你欲查公車到佗位`
- `如果你要查失物→若是你欲查失物`
- 同步修正既有句級：`如果你要補票，先去窗口抽號碼牌。→若是你欲補票，先去窗口提號碼牌。`

**回歸修正：**
- `medical`、`bus`、`transport` 多筆 `若是你要...` 改為 `若是你欲...`

### 2026-04-11：醫療句型 modal 修正（round132）
**修正句級補丁（1 條）：**
- `如果你要改看診時間，請先打電話。→若是你欲改看診時間，請先敲電話。`
- 原因：此處 `要` 表意願，應譯為 `欲`，不是保留 `要`

### 2026-04-11：診所改期／改掛流程補強（round131）
**新增詞條（2 條）：**
- `門診改到明天早上→門診改做明仔早起`
- `改掛別的醫生→改掛別位醫生`

**醫療迴歸擴充：**
- `run_medical_regression.py / doctor_flow` +4
- 新增覆蓋：臨時請假改到明早、下午門診改期、改掛別的醫生、因不方便改掛別位醫生

### 2026-04-11：診所改期／休診通知補強（round130）
**停用不一致 seed（1 條）：**
- `下禮拜→後禮拜`（與既有 `下週` / `下個星期` → `下禮拜` 的方向衝突）

**新增句級補丁（3 條）：**
- `這位醫生今天下午休診。→這位醫生今仔日下晝無看診。`
- `今天停診，請你改天再來。→今仔日無看診，請你改工閣來。`
- `如果你要改看診時間，請先打電話。→若是你要改看診時間，請先敲電話。`

**醫療迴歸擴充：**
- `run_medical_regression.py / doctor_flow` +4
- 新增覆蓋：下禮拜回診、下午休診、停診通知、改看診時間先打電話

### 2026-04-11：診所／門診流程補強（round129）
**新增詞條（2 條）：**
- `下星期→下禮拜`
- 句級補丁：`看完診再去櫃台拿藥單。→看完診閣去櫃檯提藥單。`

**醫療迴歸擴充：**
- `run_medical_regression.py / doctor_flow` +4
- 新增覆蓋：下星期回診、門診提早結束、看完診拿藥單、醫生未到外面等候

### 2026-04-11：診所／掛號情境擴充（round128）
**新增詞條（6 條）：**
- `診所→醫生館`、`櫃台→櫃檯`、`交給→交予`、`看到幾點→看甲幾點`
- 句級補丁：`第一次來診所要帶健保卡。→頭擺來醫生館愛帶健保卡。`
- 句級補丁：`如果過號了，要再去櫃台處理嗎？→若是過號矣，愛閣去櫃檯處理無？`

**醫療迴歸擴充：**
- `run_medical_regression.py / registration` +6
- 新增覆蓋：診所掛號、初診帶卡、櫃台報到、診所看到幾點、交卡櫃台、過號處理

**備註：**
- `data/lexicon_entries.jsonl` 內已存在 round120～127 的 bus prompt cleanup；舊 `progress.md` 未完整補登，本次先把最新 round 校正到資料檔實際狀態

### 2026-04-11：親子/家庭領域開拓 + 進行式大擴充（round119）
**核心進行式擴充（core_lexicon.json，適用所有主語）：**
- `在哭→佇咧哭`、`在玩→佇咧玩`、`在煮飯→佇咧煮飯`、`在洗碗→佇咧洗碗`
- `在洗澡→佇咧洗身軀`、`在睡午覺→佇咧睏晝`、`在餵奶→佇咧飼奶`、`在工作→佇咧工作`

**Rule 擴充：**
- PRONOUN 進行式動詞 +1：煮(\u716e)（`rl_817e6efe2ce1`）
- 新增家庭名詞主語進行式規則：`阿母|阿爸|阿公|阿媽|阿弟仔|阿兄|阿姊|小妹仔|後生|老婆|翁婿 在V → X佇咧V`（`rl_b1401111d50e`）

**新增詞條（4 條）：**
- `女兒→查某囝`、`老公→翁婿`、`寶寶→囡仔`、`不聽話→毋聽喙`

**新增迴歸測試：**
- `run_family_regression.py`（20 筆，5 類：parent_child/health_care/siblings/grandparents/daily）

### 2026-04-11：學校/教育領域開拓 + 進行式規則擴充 + base seed 清查（round118）
**停用惡性 base seed（共 10 條）：**
- `視力減退→青光眼`、`椰子蟹→八卦`（語義完全錯誤）
- `顏色偏紅→紅蔥仔頭`、`行文淺易→童話`（描述→無關詞）
- `不利農耕→北歐`、`不得疏忽→六月天`（危險定義式）
- `不接受→拒絕`、`方言詞→詞彙`（語義變形）
- `一種台灣小吃→蚵仔煎`（定義式）、`袂曉→精通`（台語詞被誤轉）

**Rule 擴充：**
- 進行式動詞 +3：上(\\u4e0a)/教(\\u6559)/讀(\\u8b80)（`rl_817e6efe2ce1`）
- PRONOUN 模板加入 `逐家`（大家轉換後可觸發進行式）
- 新增學校名詞主語進行式規則：`(老師|學生|同學)在V → X佇咧V`（`rl_school_noun_progressive`）

**新增詞條（3 條）：**
- `不會說→袂曉講`、`不會做→袂曉做`、`不太會→袂啥曉`

**新增迴歸測試：**
- `run_school_regression.py`（20 筆，6 類：teacher_class/student_class/homework/exam/campus/ability）

### 2026-04-11：base seed 清查 + 多X詞補完 + 銀行/郵局領域（round117）
**停用惡性 base seed（共 17 條）：**
- 定義式詞條（src=定義，tgt=術語）：`報告自己的姓名→報名`、`停戰謀和→韓戰`、`擔頭誠重→查某人`
- 時間範圍→時辰（8 條）：`下午一點到三點→未時`、`下午三點至五點→申時`、`下午五點到七點→酉時`、`上午七點到九點→辰時`、`凌晨一點到三點→丑時`、`凌晨三點至五點→寅時`、`早上五點至七點→卯時`、`晚上七點至九點→戌時`
- 其他定義式：`違反法令的行為→犯案`、`以契約的方式→包商`、`屬於北方的區域→北區`、`屬於國家所有的→國有`、`屬於國有的道路→國道`、`屬於認識的主體→主觀`

**新增詞條（6 條）：多X疑問詞**
- `多重→偌重`、`多長→偌長`、`多遠→偌遠`、`多深→偌深`、`多快→偌緊`、`多大→偌大`

**新增迴歸測試：**
- `run_bank_regression.py`（21 筆，4 類：bank_account/bank_transaction/bank_service/postal）

### 2026-04-11：飯店/計程車領域開拓 + 位置詞修正（round116 續）
**新增詞條（4 條）：**
- `在前面→佇頭前`、`在後面→佇後壁`、`在旁邊→佇旁邊`（位置詞介系詞修正）
- `幾點要退房→幾點愛退房`（退房義務語境 `要→愛`）

**新增迴歸測試：**
- `run_hotel_regression.py`（17 筆，5 類：reservation/check_in/check_out/amenities/issues）
- `run_taxi_regression.py`（19 筆，5 類：hailing/destination/navigation/payment/misc）

### 2026-04-11：餐廳/購物領域開拓 + bug 修正（round116）
**停用詞條（共 1 條）：**
- `這件→這層`（round5 舊詞條，誤傷衣物/商品情境，改用更精確的 `這件事`/`這件事情`→`這層代誌`）

**新增詞條（5 條）：**
- `不辣→無辣`（修正口味偏好句型）
- `在點菜→佇咧叫菜`（修正進行式 + 點菜合體）
- `這件事→這層代誌`、`這件事情→這層代誌`（取代過廣的 `這件→這層`）

**新增迴歸測試：**
- `run_restaurant_regression.py`（20 筆，5 類：ordering/spice_dietary/seating/payment/service）
- `run_shopping_regression.py`（17 筆，5 類：browsing/bargaining/purchase/payment/after_sales）

### 2026-04-11：系統性錯誤批次修正（round115）
**停用的惡性詞條（共 9 條）：**
- `刷卡→鑢卡`（錯誤 manual_hotfix）、`很餓→枵燥`（錯誤 base seed）
- `三天→三對時`、`你睡得好嗎→你睏了有飽眠無`、`沒睡好→無眠`（base seed）
- `有一次→有一斗`、`三次→三改`（計量詞錯誤 seed/hotfix）
- `一個月→一月日`（base seed）、`工作→工作`（identity 保護阻擋進行式規則）

**新增詞條（15 條）：**
- `很餓→真枵`、`三天→三工`、`兩天→兩工`、`昨晚→昨暗`、`明晚→明仔暗`
- `你睡得好嗎→你有眠飽無`、`沒睡好→睏袂飽`
- `刷卡→刷卡`（identity 保護）、`快點→緊咧`、`她要→伊欲`
- `三次→三擺`、`兩次→兩擺`、`幾次→幾擺`、`有一次→有一擺`、`多次→多擺`

**Rule 擴充：**
- 進行式動詞 +3：歇（歇睏）/工（工作）/考（考慮）（`rl_817e6efe2ce1`）
- 新增樓層位置規則：`在N樓→佇N樓`（`rl_5ccd078d7efa`，priority 87）

### 2026-04-11：大批 base seed 清查 + 多類別系統修正（round114）
**停用的惡性 base seed（共 5 條）：**
- `不見`→`無去`、`下個月`→`後個月`、`很高`→`足懸的`、`跑走`→`逃走`、`完了`→`完結`

**Allowlist 修正：**
- 移除 `看電視`、`吹冷氣`（不需保護，但遮蓋後阻擋進行式規則）

**Rule 擴充：**
- 進行式動詞 +6：吹/唱/玩/跑/買/忙（`rl_817e6efe2ce1`）
- 新增 fluency rule：`欲說` → `欲講`、`咧說` 已有

**新增詞條（28 條）：**
- `不清楚`/`不明白`/`不記得`/`不確定`/`不太確定` 系列
- `會說`/`欲說`/`想說` → X講
- `睡不著`→`睏袂去`、`上週`/`下週`/`這週`→禮拜系列
- `外面在下雨`/`外面下雨了`、`上個星期`/`下個星期`
- `牙痛`→`牙疼`、`喉嚨痛`→`嚨喉疼`、`喉嚨`→`嚨喉`
- `見到`→`見著`、`找到`→`揣著`、`跟著`→`綴`
- `不願意`→`毋肯`、`不應該`→`毋著`
- `厲害`→`利害`、`好厲害`→`真利害`、`跑走`→`走去`

### 2026-04-11：日常會話 bug 修正 + 薄弱迴歸補強（round113）
- 停用惡性 base seed：`不見`→`無去`（修正「好久不見」→「好久無去」的錯誤）
- 新增 4 條 round113 詞條：`好久不見`→`好久無見`、`你辛苦了`→`你辛苦矣`（修正重複「你」）、`不確定`→`無確定`、`不太確定`→`無啥確定`
- 新增 `run_conversation_regression.py`（30 筆，5 類：greetings/status_check/daily_chat/daily_response/schedule_plans）
- medical 補強：doctor_flow 6→10、tests 6→10（共 43→51）
- transport 補強：crowd_safety 5→8（共 57→60）

### 2026-04-10：base seed 修正 + 薄弱迴歸類別補強（round112）
- 停用 3 條嚴重錯誤的 base seed：`你要注意`→`笑面虎`、`探視`→`探頭`、`住院`→`入院`
- 新增：`右手邊`→`正手爿`、`左手邊`→`倒手爿`、`住院`→`蹛院`、`住院櫃檯` identity 保護、`探視` identity 保護
- 迴歸測試補強：medical 30→43（rooms_inpatient 2→8、pharmacy_payment 3→7、redirect 3→6）
- transport 52→57（destinations 3→8）

### 2026-04-07：說->講 大規模修正（e5fbd61）
- 說->講 完整鏈：代名詞/在說/咧說/常見主語/說看
- 位置介系詞：在[台灣主要城市]->佇、住在->蹛佇
- 進行式 verb 清單擴充：等/聽/學/哭
- 停用 8 條嚴重錯誤詞條（你說->外人、不說->不論 等）
- post_cleanup 修正：懸雄->高雄、懸速公路->高速公路

### 2026-04-03～04：日常會話 round107–111
收到、了解、沒事、辛苦了、麻煩你了、掰掰、早點休息、等一下打給你、到哪了、剛到家、出發了沒、要轉車嗎 等 ~50 條日常口語短語

---

## 覆蓋率仍薄的地方

### 迴歸測試內（量少、遇新句型容易破）
- `shopping / bargaining`：已 4 條，可補殺價、湊整、買多折扣、贈品
- `shopping / after_sales`：已 6 條，可補退款進度、換貨、保固、物流延遲
- `school / student_class` 與 `school / teacher_class`：各 4 條，可補請假、遲到、點名、家長聯絡

### 尚無迴歸測試的情境（已知會踩雷的領域）
- 診所/掛號情境（擴充版）—— 已有部分，可持續補現場排隊、改掛與過號後續
- 線上購物售後 —— 目前只有基本購物 regression，訂單/退款/物流仍可補強

---

## 下一步優先工作

1. **繼續補薄弱迴歸類別**
   - `shopping / bargaining`：可補到 8～10
   - `shopping / after_sales`：可補到 10～12
   - `school / student_class`、`school / teacher_class`：可各補到 8～10

2. **購物售後/議價與學校班級情境補充**
   - `shopping / bargaining` 可補殺價、湊整、買多折扣、贈品
   - `shopping / after_sales` 可補退款進度、換貨、保固、物流延遲
   - `school / student_class`、`school / teacher_class` 可補請假、遲到、點名、家長聯絡
   - 先用實測輸出找華語殘留，再補人工詞條與 regression

3. **已知 edge case（低優先）**
   - 無主語句 `要V` → `要` 不轉義務 `愛`（如 `要走高速嗎` → `要走懸速無`）

---

## 快速定位問題的方法

```bash
# 某句輸出不對，先確認哪個詞條命中
python3 app.py --trace "問題句子"

# 搜尋可能 shadow 的舊詞條
grep -n "關鍵字" data/lexicon_entries.jsonl | grep '"status":"active"'

# 確認目前最新 round
grep -o '"source":"curation:round[0-9]*' data/lexicon_entries.jsonl | grep -o 'round[0-9]*' | sort -t d -k 2 -n | tail -3
```
