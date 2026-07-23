# Local data layout

入力データはGitに追加されません。次のように配置するか、各コマンドの引数で別パスを指定してください。

```text
data/
├─ all_roi_spectra200x200.csv
├─ all_map_spectra.csv                 # 広域診断を行う場合のみ
├─ ch4_lut.csv
├─ qa/
│  └─ HISUI QA_IM / QA_DM TIFF files
└─ 2200nm/
   ├─ reference_plume_mask.npy
   ├─ qa_guided_outputs/
   │  ├─ raw_mf.npy
   │  └─ recommended_alpha_corrected.npy
   └─ mf_outputs/
      └─ median_thin_eachiter_then_broad_median_plume_mask.npy
```

CSV、TIFF、NumPy配列などの元データと中間生成物は `.gitignore` で除外されています。
