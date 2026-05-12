◎Monocular Gait Prediction Research

・概要
　本研究は、単眼のRGBカメラ映像から人の歩行を三次元の下肢骨格系列として取得し、その時系列情報を用いて将来の歩行状態を予測することを目的とする  
高価なモーションキャプチャ装置を使わず、一般的なカメラのみで歩行を定量的に解析し、さらに少し先の動きまで予測できる仕組みの構築を目指す

具体的には、動画から左右の股関節・膝・足首の三次元座標を取得し、床面補正、骨長補正、接地判定、支持脚推定などの前処理を行ったうえで、LSTMベースの時系列モデルにより未来の骨格位置と支持脚状態を予測する
評価では、位置誤差だけでなく、Z方向誤差、骨格形状の保持、支持脚予測精度なども確認し、可視化結果と合わせて歩行の自然さを検証した

・このリポジトリについて
　このリポジトリは、研究で使用した主要コードを整理したものである  
研究全体には多くの中間生成物、大容量データ、派生実験結果が含まれるが、本リポジトリでは以下を中心に掲載している

1.主要な研究コード
2.実行コマンド例
3.研究の流れが分かる説明
4.必要に応じて代表的な結果画像・GIF・CSV

生データ、元動画、全出力結果、巨大な中間生成物は含めていない

・研究の流れ
　本研究の処理は、大きく次の流れで構成されている

1. 動画から下肢6点（左右の股関節・膝・足首）の三次元座標を取得
2. 床面補正、骨長補正、接地判定、支持脚推定などを行い、学習用 *_processed.csv を生成  
3. 3DビューやYZビュー、時系列図で前処理結果を確認
4. 微小ノイズ、低周波ドリフト、時間伸縮によるデータ拡張を実施  
5. LSTMベースのモデルで未来の骨格位置と支持脚状態を予測  
6. 位置誤差、Z方向誤差、骨格形状の保持、支持脚精度などを評価し、図表やGIFを出力  

・ディレクトリ構成

├─ README.md
├─ .gitignore
├─ requirements.txt
├─ src
│   ├─ zcap5-2.py
│   ├─ zcap_draw3d.py
│   ├─ augment_processed.py
│   ├─ zmodel5.py
│   └─ eval_zmodel3.py
├─ commands
│   └─ 実行コマンド集.txt
├─ results
│   ├─ preprocessing
│   └─ evaluation
└─ docs

・各スクリプトの役割

src/zcap5-2.py
動画から下肢6点の三次元骨格を取得し、床面補正、骨長補正、接地判定、支持脚推定などを行って、学習用の *_processed.csv を生成する前処理

src/zcap_draw3d.py
前処理後の骨格データを 3D / YZ ビューや時系列図で可視化し、データの妥当性を確認する 
前処理結果の品質確認や、骨格の動きの把握に使用

src/augment_processed.py
processed.csv に対してデータ拡張を行う  
微小ノイズ、低周波ドリフト、時間伸縮などに対応し、学習データの多様化を行う

src/zmodel5.py
前処理済みの歩行データを入力として、LSTMにより未来の骨格位置と支持脚状態を予測する  
複数被験者学習、単一被験者特化学習、予測長変更などに対応

src/eval_zmodel3.py
学習済みモデルの予測結果を評価し、CSV・PNG・GIFなどを出力する  
位置誤差だけでなく、骨格形状の一致や支持脚予測精度も確認できる

・実行環境
本リポジトリは Python を用いた

・使用ライブラリ
- Python 3.x
- NumPy
- pandas
- matplotlib
- OpenCV
- MediaPipe
- SciPy
- PyTorch
- scikit-learn
- joblib
- imageio
- Pillow

・セットアップ例
powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

・実行例
1. 動画から前処理データを生成
python src/zcap5-2.py --input_glob "C:\mypro\videos\*.mp4" --out_dir outputs_final --debug_stages

2. 前処理データの確認
python src/zcap_draw3d.py --csv_glob "outputs_hitobetu_riku/*_processed.csv" --out_dir outputs_hitobetu_riku/qa --views 3d yz --fps 5

3. データ拡張（微小ノイズ）
python src/augment_processed.py --in_glob "outputs_jinbutu/*_processed.csv" --out_dir outputs_jinbutu_A --n 1 --modes A --noise_scale 0.5 --x_scale 0.1 --y_scale 0.5 --z_scale 1.0

4. データ拡張（低周波ドリフト）
python src/augment_processed.py --in_glob "outputs_jinbutu/*_processed.csv" --out_dir outputs_jinbutu_B --n 1 --modes B --drift_scale 1.0 --drift_win 81 --drift_cap_k 2.5 --x_scale 0.05 --y_scale 0.3 --z_scale 1.0

5. データ拡張（時間伸縮）
python src/augment_processed.py --in_glob "outputs_jinbutu/*_processed.csv" --out_dir outputs_jinbutu_D --n 1 --modes D --time_warp_rate 0.03

6. 学習（複数被験者）
python src/zmodel5.py --csv_glob "outputs_hitobetu/*_processed.csv" --out_dir outputs_hitobetu --val_csv "outputs_hitobetu_riku/data_riku_val*_processed.csv" --test_csv "outputs_hitobetu_riku/data_riku_test*_processed.csv" --hidden 128 --layers 1 --dropout 0.3 --batch_size 128 --lr 5e-5 --weight_decay 5e-4 --patience 20 --lambda_contact 3.0 --support_class_weight_auto --support_smooth_min_run 3 --eval_test_end --horizon 10

7. 学習（単一被験者特化）
python src/zmodel5.py --csv_glob "outputs_jinbutu/*_processed.csv" --out_dir outputs_jinbutu --val_csv "outputs_jinbutu/data_riku_val*_processed.csv" --test_csv "outputs_jinbutu/data_riku_test*_processed.csv" --hidden 128 --layers 1 --dropout 0.3 --batch_size 128 --lr 5e-5 --weight_decay 5e-4 --patience 20 --lambda_contact 3.0 --support_class_weight_auto --support_smooth_min_run 3 --eval_test_end --horizon 10

8. 評価
python src/eval_zmodel3.py --out_dir outputs_hitobetu --which test --plot_losses --plot_sample_ts --gif --view yz --gif_fps 5 --support_min_run 3 --event_win 2 --z_tol 0.05 --run_tag A_tes

・出力の例
本研究では、処理段階に応じて以下のような出力を生成する

- *_processed.csv：前処理済みの下肢骨格時系列データ

- 可視化GIF：3Dビュー、YZビューなどの骨格可視化

- 時系列図：関節位置や比較結果の可視化

- 評価CSV：各種誤差指標の集計結果

- 評価図：horizonごとの誤差推移、支持脚予測精度推移など

・この研究で重視した点
本研究では、単に予測誤差を小さくすることだけでなく、以下の点を重視した

- 単眼RGBカメラのみで扱えること
- 下肢の三次元骨格系列として歩行を表現すること
- 現在の姿勢推定だけでなく、未来予測まで扱うこと
- 接地状態や支持脚状態など、歩行らしさに関わる情報も扱うこと
- 数値指標だけでなく、可視化によって挙動も確認すること

・今後の改善課題

- Z方向推定の安定化
- 接地・支持脚判定の精度向上
- 長期ロールアウト時の予測崩れの抑制
- 前進量推定の改善
- 実歩行との整合性評価の強化

・研究の意義
本研究は、一般的な単眼RGBカメラだけを用いて、歩行を三次元的に解析し、さらに将来の歩行状態まで予測する点に意義がある
認識だけでなく予測まで含めることで、医療・福祉・見守り・ロボット応用につながる歩行解析基盤の構築を目指した

・制限事項
- 元動画データは公開していません
- 大規模な中間生成物や全実験出力は含めていません
- 学習済みモデル重みは容量と整理の都合上、同梱していません
- 単眼推定であるため、特にZ方向（奥行き）の精度には課題があります
- 実歩行距離に対して、推定された前進量が小さくなる傾向があります
- この提出用リポジトリは、研究内容と実装構成を示すことを主目的としており、研究当時の全環境・全出力の完全再現を保証するものではありません
