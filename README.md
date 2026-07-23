# HISUI methane Iterative MF (1600 + 2200 nm)

HISUIハイパースペクトルデータを対象に、1600 nm帯のIterative Matched Filter、QAに基づく方向性ストライプ除去、2200 nm帯との融合判定を行う研究用コードです。

1600 nmと2200 nmではMFの感度とノイズ分散が異なるため、αを直接加算しません。各帯域をmedian/MADでrobust z-score化し、両帯域の正の応答と空間連結性が一致する領域をメタン候補として抽出します。

![Dual-band methane detection](docs/figures/dual_band_detection_result.png)

## 主な処理

1. 1580–1700 nmからCH₄ unit absorption spectrumを作成
2. Iterative MFによる1600 nm αマップの推定
3. QA_DM・CTメタデータに基づく方向性ストライプ除去
4. 2200 nm結果に残った細線の追加補正
5. 1600/2200 nmのrobust z-score融合
6. 負側tail、空間シフト、既知地表線による偽陽性検査

## フォルダ構成

```text
.
├─ scripts/        解析スクリプト
├─ tests/          合成データを用いた単体テスト
├─ data/           ローカル入力データ（Git管理対象外）
├─ outputs/        実行結果（Git管理対象外）
└─ docs/figures/   公開用の代表結果図
```

## セットアップ

Python 3.10以降を想定しています。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 入力データ

元データは容量と配布条件のため、このリポジトリには含まれていません。配置方法と必要なファイル名は [data/README.md](data/README.md) を参照してください。

ROI CSVは次の列を持つ形式です。

```text
y,x,wave_485.00nm,...,wave_1687.89nm,...
```

CH₄ LUT CSVは、波長列と複数の濃度・enhancement列を持つ形式です。波長はµmまたはnmを自動判別します。

## 実行例

### 1600 nm Iterative MF

```powershell
python scripts/iterative_mf_1600nm.py `
  --roi-csv data/all_roi_spectra200x200.csv `
  --modtran-csv data/ch4_lut.csv `
  --output-dir outputs/iterative_mf_1600nm
```

### QA傾き解析とストライプ除去

```powershell
python scripts/analyze_qa_slope_1600nm.py --qa-dir data/qa

python scripts/refine_2200nm_residual_line.py `
  --qa-output-dir data/2200nm/qa_guided_outputs `
  --mf-output-dir data/2200nm/mf_outputs

python scripts/iterative_mf_1600nm_destriped.py `
  --roi-csv data/all_roi_spectra200x200.csv `
  --modtran-csv data/ch4_lut.csv `
  --reference-2200-mask data/2200nm/reference_plume_mask.npy
```

### 1600/2200 nm融合判定

```powershell
python scripts/fuse_1600_2200_methane_detection.py
```

既定条件では、両帯域3 robust σ以上かつ相関補正joint zが4以上をcoreとし、両帯域2σ以上までextentを領域成長します。入力・出力パスや閾値は各スクリプトの `--help` で変更できます。

## テスト

```powershell
python -m unittest discover -s tests -v
```

## 今回の200×200 ROIでの結果

|候補|core / extent|ROI座標 (y, x)|joint z 最大|
|---|---:|---|---:|
|R1|26 / 53画素|y=79–90, x=42–53|8.729|
|R2|22 / 70画素|y=96–106, x=95–111|9.653|

空間シフト1000回の帰無試験で、実測の連結core 48画素以上になった例は0回でした。負側3σでは連結成分0、既知の右下地表線も最終core・extentとも0画素でした。詳しい診断図は [dual_band_detection_diagnostics.png](docs/figures/dual_band_detection_diagnostics.png) にあります。

## 注意

この判定は「両吸収帯で正のMF応答が空間的に一致するメタン適合領域」を示します。メタン排出の確定には、風向・風速、設備位置、別時刻または別センサとの照合が必要です。RGB上の明るい地表構造に重なる候補は、surface spectral artifactの可能性も残ります。

## 謝辞

基本的なIterative MFの構成は [dora-743/MF](https://github.com/dora-743/MF/tree/main/IterativeMF) を参照しています。

## License

ライセンスはまだ指定していません。再利用・再配布についてはリポジトリ所有者へ確認してください。
