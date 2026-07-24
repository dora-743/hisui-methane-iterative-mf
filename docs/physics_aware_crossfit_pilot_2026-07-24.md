# HISUI 物理テンプレート＋波長 cross-fit パイロット

実施日: 2026-07-24

## 目的

MODTRAN の CH4 濃度系列から HISUI 分解能の吸収テンプレートを作り、
1.58–1.75 µm と 2.20–2.39 µm を相互に selection / validation に使う
cross-fitted likelihood ratio を試した。通常の strong / weak / combined MF、
人工注入、逆符号 CH4 対照、e-BH も同じ処理で比較した。

ここで出力する e-value は Gaussian 背景の plug-in 値である。平均・共分散を
外部データで固定した理想条件では条件付き期待値が1以下になるが、同じシーンから
背景を推定する現在の実装は、その有限標本保証を持たない。結果ではこの点を
候補ランキングと FDR 保証の違いとして明示する。

## 入力

- ROI: `D:\research\code\all_roi_spectra200x200.csv`
  - 40,000画素、200×200、185バンド、選択帯域では欠損なし
- 全体シーン: `E:\refit\all_map_spectra.csv`
  - 3,335,334行、座標範囲1894×1761、選択帯域で有効1,562,452画素
- MODTRAN: `E:\refit\CH4a.csv`
  - 380–2500 nm、濃度パラメータ0–5（0.1刻み）

MODTRAN 放射輝度の100倍係数は絶対値比較には必要だが、
`-d log(radiance) / d concentration` で作る unit absorption spectrum では
厳密に相殺される。100倍後の MODTRAN に対するシーン中央値の比は、最終設定で
ROI が weak 0.463 / strong 0.277、全体が weak 0.502 / strong 0.311 だった。
これは単純な単位差だけでなく、地表・照明・大気条件の差が残ることを示す。

## 実装したもの

- Gaussian SRF（FWHM 12.5 nm）による MODTRAN→HISUI リサンプリング
- log-radiance UAS と通常 MF（strong / weak / combined）
- strong で非負振幅を選び weak で検証する LR と、その逆方向 LR の平均
- 5分割の空間 block nuisance cross-fitting
- 各吸収帯で定数＋一次 continuum を除去
- 全体シーン用の250×250局所背景モデル（周囲1タイルを学習に含める）
- e-BH、逆符号 CH4 対照、正逆の clipped pooled-tail 診断
- MODTRAN 比率による半人工注入ベンチマーク
- チャンク読み込み、候補成分の RGB / log-e / 局所スペクトル図

2.40 µm 端の2399.82 nmバンドは暗い水面や走査方向の細線を強く拾った。
上限を1バンド下の2.390 µmにすると ROI の逆符号 e-BH が0になったため、
これを既定値とした。ただし全体シーンには別の端・境界アーティファクトが残る。

## 主要結果

### テンプレート族

濃度だけを変えた正規化テンプレート族は99%有効ランク1、99.9%でランク2、
第1成分のエネルギー比0.99772、隣接濃度テンプレートの最小 cosine 0.999943
だった。したがって、この LUT はほぼ「同じ形状の振幅違い」であり、濃度だけでは
surface confuser を退ける高ランクの物理 manifold にならない。

### 人工注入感度

表は実背景画素へ MODTRAN の厳密な log-radiance 比を加え、各手法の経験的
FPR 0.1% における TPR を測ったもの。`cross-fit` は双方向平均 log-e である。

ROI:

| 濃度 | strong MF | combined MF | weak MF | dual-min | cross-fit |
|---:|---:|---:|---:|---:|---:|
| 0.1 | 0.0057 | 0.0102 | 0.0054 | 0.0359 | 0.0165 |
| 0.2 | 0.1176 | 0.2633 | 0.0225 | 0.1855 | 0.3064 |
| 0.5 | 0.9995 | 0.9998 | 0.3462 | 0.7779 | 0.9928 |
| 1.0 | 1.0000 | 1.0000 | 0.9621 | 0.9977 | 1.0000 |

全体シーンサンプル:

| 濃度 | strong MF | combined MF | weak MF | dual-min | cross-fit |
|---:|---:|---:|---:|---:|---:|
| 0.1 | 0.0029 | 0.0038 | 0.0013 | 0.0355 | 0.0127 |
| 0.2 | 0.0117 | 0.0274 | 0.0015 | 0.1652 | 0.3177 |
| 0.5 | 0.9932 | 0.9981 | 0.0036 | 0.7108 | 0.9833 |
| 1.0 | 0.9999 | 1.0000 | 0.0810 | 0.9898 | 0.9996 |

濃度0.2では cross-fit が通常 MF より良く、異なる吸収帯の整合性を使う利点が
見えた。一方、濃度0.5以上では combined MF もほぼ飽和しており、cross-fit の
主目的は感度向上より confuser 排除と selection-safe な検証にある。

### 実シーンと負の対照

ROI の最終2.390 µm設定では、正方向・逆符号とも e-BH（目標FDR 10%）は0。
事前に注目していた R2（y=96–106, x=95–111）は最大 log-e 11.52で全40,000画素中
3位、log-e 5以上が11画素あり、候補ランキングとしては残った。R1 は最大2.52、
110位だった。R2 は「有望候補」ではあるが、この実験だけで検出確定とはしない。

全体シーンでは局所背景化後も、正方向 e-BH が252画素、逆符号が109画素だった。
上位成分には暗い貯水池、明るい施設、シーン境界に沿う細線が含まれ、正方向だけを
メタンと解釈できない。plug-in e の log-mean は正301.7、逆308.1で、理想的な
e-value の期待値1から大幅に外れた。

正・逆を20で clip して pooled mean を1へ正規化する保守的な裾診断では、
ROI・全体とも正方向/逆方向の e-BH は0だった。この正逆校正自体も sign
exchangeability が必要な診断であり、厳密な保証ではないが、現在のデータから
「FDR保証付き発見あり」と主張すべきでないことは明確である。

## 結論

1. 波長 cross-fit は半人工注入で機能し、特に中程度の注入で通常 MF より高い
   検出率を示した。
2. R2 は狭いシーン内で上位に残り、追加検証に値する。
3. 濃度だけの MODTRAN sweep は実質ランク1で、研究案が必要とする物理 target
   family としては不足する。
4. 全体シーンの Gaussian plug-in e-value は重い裾と空間的不均一性で破綻した。
   現時点の252画素をメタン発見として報告してはいけない。

## 次に行う実験

優先順位は次の通り。

1. 別 HISUI シーンまたは独立背景領域で平均・共分散と裾校正を固定し、対象シーンを
   完全 held-out にする。これが e-value / e-BH の保証に最も直接的。
2. H2O、surface albedo、aerosol、solar/view geometry を振った MODTRAN grid を
   作り、濃度以外の形状変化を含む有効ランク2以上の target family を構成する。
3. land-cover cluster または局所 multivariate-t / mixture 背景を比較し、逆符号
   対照の e-BH が0または名目水準内になるまでモデルを選ぶ。
4. pixel を仮説単位にせず、事前定義した空間 block / connected component を
   仮説にして多重性を下げ、block conformal 校正を試す。
5. R2 を風向、設備位置、別時刻・別センサーと照合する。これらの外部情報なしに
   plume と断定しない。

## 再現コマンド

```powershell
python scripts/physics_aware_crossfit_mf.py `
  --scene-csv "D:\research\code\all_roi_spectra200x200.csv" `
  --modtran-csv "E:\refit\CH4a.csv" `
  --output-dir outputs/crossfit_final_roi200 `
  --spatial-block-size 10

python scripts/physics_aware_crossfit_mf.py `
  --scene-csv "E:\refit\all_map_spectra.csv" `
  --modtran-csv "E:\refit\CH4a.csv" `
  --output-dir outputs/crossfit_final_full_scene `
  --spatial-block-size 20 `
  --local-tile-size 250 `
  --local-tile-halo 1 `
  --chunksize 100000

python scripts/inspect_crossfit_candidates.py `
  --scene-csv "E:\refit\all_map_spectra.csv" `
  --analysis-dir outputs/crossfit_final_full_scene
```

生成される `outputs/` は入力データと同様に Git 対象外である。再現に必要なコード、
テスト、設定、数値報告だけをリポジトリへ保存する。
