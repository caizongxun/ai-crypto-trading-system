# Market Regime-Based Trading System

## 核心理念改變

### 舊系統問題
原本的三分類模型(Long/Short/No-trade)直接預測進場方向,導致:
- 18537筆交易但只有26.53%勝率
- 模型學到的是「什麼時候會觸發標籤」而非「什麼時候進場會贏」
- 過度擬合訓練期的市場特徵

### 新系統架構

**關鍵改變: 分離預測與決策**

```
舊: 特徵 → 模型 → 進場信號(Long/Short/None)
新: 特徵 → 模型 → 市場狀態(5種regime) → 規則系統 → 進場信號
```

## 5種市場狀態(Regime)

模型會將每根K棒分類到以下5種狀態之一:

### Regime 0: 強勢上升趨勢
- **特徵**: ADX > 25, 價格在所有EMA之上, MACD正向
- **進場規則**: 等價格回調到EMA9/21時做多
- **條件**: RSI 30-60(未超買), 成交量正常, 不在BB上緣
- **停損/停利**: 2.0 ATR / 4.0 ATR (大目標)

### Regime 1: 強勢下降趨勢
- **特徵**: ADX > 25, 價格在所有EMA之下, MACD負向
- **進場規則**: 等價格反彈到EMA9/21時做空
- **條件**: RSI 40-70(未超賣), 成交量正常, 不在BB下緣
- **停損/停利**: 2.0 ATR / 4.0 ATR

### Regime 2: 震盪盤整
- **特徵**: ADX < 25, 價格在EMA上下震盪
- **進場規則**: **不交易**(勝率太低)

### Regime 3: 高波動突破
- **特徵**: BB寬度擴張, 成交量暴增, 價格突破BB
- **進場規則**: 順突破方向進場
- **條件**: 成交量 > 2倍均量, ADX上升, MACD同向
- **停損/停利**: 1.0 ATR / 2.0 ATR (快進快出)

### Regime 4: 低波動整理
- **特徵**: BB寬度收縮, 成交量低迷
- **進場規則**: **不交易**(等突破)

## 為什麼這樣設計?

### 1. 避免過度擬合
模型只需要學「當前市場處於什麼狀態」,這比「現在該做多還是做空」更穩定。市場狀態是客觀的,但最佳進場點是主觀且動態的。

### 2. 可解釋性
每筆交易都有明確理由:
- "Regime 0: Uptrend pullback to EMA"
- "Regime 3: High vol breakout up"

你可以回測後看到「哪種regime表現最好」,然後只交易那幾種。

### 3. 彈性調整
不喜歡某個regime的表現? 直接在`rule_based_entry.py`把它改成return 0 (不交易)即可,不用重新訓練模型。

### 4. 不同regime用不同策略
- 趨勢regime用大停損大目標(追求趨勢利潤)
- 突破regime用小停損快目標(避免假突破)
- 震盪regime直接不交易

## 使用方式

### 1. 啟動新版GUI
```bash
streamlit run app_regime.py
```

### 2. 訓練Regime分類器
- 選擇幣種與時間週期
- 點擊"Train Regime Classifier"
- 系統會用KMeans自動標註5種regime
- 然後訓練XGBoost分類器

### 3. 回測
- 調整風險參數(槓桿, 每筆風險%)
- 調整"Min Regime Confidence"(regime預測信心閾值)
- 點擊"Run Regime Backtest"
- 查看"Performance by Regime"表格,找出哪種regime最賺錢

### 4. 優化策略

**方法1: 調整信心閾值**
- 提高"Min Regime Confidence"會減少交易數但提高質量
- 0.5 = 模型至少50%信心才進場

**方法2: 禁用特定regime**
編輯`src/rule_based_entry.py`,把表現差的regime改成:
```python
def _choppy_logic(self, row, confidence):
    return {'signal': 0, 'regime': 2, ...}  # 永遠不交易
```

**方法3: 調整進場條件**
修改各個`_xxx_logic`函數中的條件:
- 放寬條件 = 更多交易
- 嚴格條件 = 更高勝率

**方法4: 調整停損停利**
在`calculate_stop_loss`和`calculate_take_profit`中修改ATR倍數。

## 預期表現

相比舊系統(18537筆, 26%勝率),新系統應該會:
- **交易數大幅減少**(因為regime 2和4不交易)
- **勝率提升**(只交易高機率regime)
- **每筆交易更有邏輯**

假設回測結果:
- Regime 0: 200筆, 65%勝率 → 保留
- Regime 1: 180筆, 62%勝率 → 保留
- Regime 2: 0筆 → 已禁用
- Regime 3: 150筆, 55%勝率 → 可保留或調整
- Regime 4: 0筆 → 已禁用

總計: 530筆, 預期勝率60%+

## 進一步優化方向

### 1. 只交易最佳regime
回測後如果發現Regime 0表現最好,就只開啟它:
```python
if regime in [1, 2, 3, 4]:
    return {'signal': 0, ...}  # 只留regime 0
```

### 2. 根據時段過濾
某些時段(如亞洲盤)某個regime可能表現更好,可以加時間過濾。

### 3. 多幣種組合
對BTCUSDT, ETHUSDT, SOLUSDT各訓練一個regime分類器,分散風險。

### 4. 動態調整參數
根據近期表現動態調整各regime的停損停利倍數。

## 文件結構

```
src/
├── regime_classifier.py      # Regime分類模型
├── rule_based_entry.py       # 各regime的進場規則
├── regime_backtester.py      # Regime系統回測引擎
├── feature_engineering.py    # 特徵工程(共用)
└── data_loader.py            # 數據載入(共用)

app_regime.py                 # Regime系統GUI
README_REGIME.md             # 本文件
```

## 核心優勢總結

| 項目 | 舊系統 | 新系統 |
|------|--------|--------|
| 預測目標 | 做多/做空/不做 | 市場狀態 |
| 進場決策 | 模型直接給 | 規則系統根據狀態決定 |
| 可解釋性 | 黑箱 | 每筆交易有理由 |
| 調整彈性 | 需重訓模型 | 改規則即可 |
| 風險管理 | 統一參數 | 各regime不同參數 |
| 過擬合風險 | 高 | 低 |

## 開始使用

1. `streamlit run app_regime.py`
2. 訓練BTCUSDT 15m的regime分類器
3. 回測看各regime表現
4. 調整規則或禁用表現差的regime
5. 重新回測驗證

祝交易順利!
