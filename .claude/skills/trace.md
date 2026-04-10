---
name: trace
description: 對一句話同時跑 --trace 和 --explain，診斷轉換結果
---

使用者要診斷的句子是：$ARGUMENTS

請依序執行以下兩個指令，並完整輸出結果：

```bash
python3 app.py --trace "$ARGUMENTS"
python3 app.py --explain "$ARGUMENTS"
```

輸出結果後，根據 trace JSON 和 explain 內容：

1. **指出最終輸出**（轉換後的台語漢字）
2. **找出關鍵詞條**：哪些 phrase/sentence 命中了，各自的 src → tgt 是什麼
3. **標示可疑點**：若有詞條沒有命中、或輸出看起來不對，指出可能的原因（例如：被舊 identity entry shadow、沒有對應詞條、rule 優先順序問題）
4. **建議下一步**：是否需要新增詞條、停用舊詞條，或調整優先順序

若使用者沒有提供句子，請先詢問要診斷哪一句。
