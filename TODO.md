# sharedimp-py 作業記録と TODO

最終更新: 2026-11-13（R/Python の仕様調整完了時点。直前の `main` は `c082026`）

このファイルは作業の引き継ぎ用メモです。仕様そのものは
`../sharedimp-r/docs/specification.md`（R/Python 共通仕様の正本）、Python 側の適合状況は
`docs/specification.md`、R 側への申し送りは `docs/r_side_changes.md` を参照してください。

## 1. 現在の状態

| 項目 | 状態 |
| --- | --- |
| ブランチ | `main`（push 済み） |
| pytest（scikit-learn 1.9.1） | 212 passed, 1 skipped（scikit-learn 側の array-API チェック） |
| pytest（最小依存: scikit-learn 1.3.2 / numpy 1.23.5 / pandas 1.5.3） | 201 passed |
| ruff / mypy / build / `twine check --strict` | すべて合格 |
| GitHub Actions `CI` | 成功（Python 3.10–3.13、macOS、Windows、最小依存、build） |
| GitHub Actions `DockerHub/GHCR` | 成功。`main` への push ごとに `latest` と短縮 SHA を公開 |
| Docker イメージ | `ghcr.io/dsc-chiba-u/sharedimp-py`、`koki/sharedimp-py`（同一 digest） |
| PyPI | **未公開** |
| R 側で確認した版 | `sharedimp-r` `cf2e1cc`（R 参照値は `92f9ecb` から生成。以降は文書のみの変更。R 側には一切書き込んでいない） |

## 2. これまでの作業

### 2.1 初期実装（`dd7c66a`〜`bfcb2a5`）

- `shared_importance()`: fold 内 OLS 残差化、Full/Reduced の paired 評価、任意の
  scikit-learn estimator・scorer・CV splitter、`n_jobs`、`random_state`、負の importance を保持。
- `AnchorResidualizer`: 残差化を独立した scikit-learn transformer として実装
  （scikit-learn の estimator チェックを通過）。
- モジュール分割: `_residualizer` / `_resampling` / `_fitting` / `_scoring` / `_result` / `_core`。
- 公式 UMFI 実装（`guanton/umfi_python`、固定コミット）との parity test。
  ライセンスがないため同梱せず、テスト時に取得して SHA-256 で検証。
- パッケージ構成、CI、README、CHANGELOG、CITATION.cff、examples。

### 2.2 R 版との共通仕様統一（`febd7ea`〜`1bb2ce8`）

- Direction diagnostic を公開 API から削除（`_direction.py` は非公開の experimental として残置）。
- Full 表現で anchor を元の列位置に保つ仕様を確定。
- 完全共線: `SS_res ≤ 1e-10 × SS_tot` で残差をゼロ固定、列は残す。
  定数列は `max == min` で厳密判定（平均の丸め誤差で定数 anchor を見逃す問題を修正）。
- `p = 1` を許容。Reduced は null model（回帰: training mean、分類: training class frequencies）。
- 出力 schema: summary 6列、fold 表は split 優先の行順・Python は 0 始まり。
- `common_scorer()`: `mse, rmse, mae, r2, rsq, log_loss, accuracy, roc_auc` の定義を
  R (yardstick) と一致させた scorer。`r2` = `rsq_trad`、`rsq` は相関の二乗。
- cross-language parity テスト（R の実出力を参照値として保存）。
- 公式 UMFI parity を「完全一致が期待される条件」「期待されない条件」に分割。

### 2.3 Docker（`692c3fe`）

- `Dockerfile`（`python:3.13-slim`、2段ビルド、非 root、ビルド中に smoke test）。
- `.dockerignore`（許可リスト方式）、`docker/smoke_test.py`。
- `.github/workflows/build_test_push.yml`: PR は build/test のみ。`main` は `latest` + 短縮 SHA、
  `vX.Y.Z` タグは `X.Y.Z` + `X.Y` + 短縮 SHA。GHCR と Docker Hub に同一ビルドを push。
  Docker Hub の secrets 未設定時はビルド前に明示的なエラーで停止。
- secrets（`DOCKER_USERNAME` / `DOCKER_PASSWORD`）は登録済み。

### 2.4 共通仕様の完成（`399bb20`、`dd0bef0`）

- 既定 metric: `scoring=None` は回帰で MSE（`neg_mean_squared_error`）、
  分類で multinomial log loss（`neg_log_loss` として報告）。`estimator.score` は使わない。
- Learner failure: `on_error="warn"`（既定）で NaN を記録し、`error` 列にメッセージ、
  warning を1回出して続行、成功 fold だけで集計（`n_splits`、`result.n_failed`）。
  `on_error="error"` は即時停止。分類で training fold が1クラスの場合も失敗扱い。
- `docs/specification.md` を「R 正本へのポインタ + 適合表 + Python 固有事項 + 差異の記録」に変更。
- Docker workflow の手動実行に `publish` 入力を追加（R 版と同じ）。

### 2.5 cross-language parity の完成（`98de522`〜`c082026`）

- R 参照値を最新 R 版（`92f9ecb`）から再生成。`generate_r_reference.R` を新 schema に対応させた。
- `tests/test_cross_language.py` から旧 R 実装向けの読み替え（符号反転、列の並べ替え、
  行順変更）を削除し、R の出力と直接比較する形にした（split 番号の 0/1 始まりだけは仕様どおり）。
- 比較対象: 残差化係数、Full/Reduced 行列、予測値、fold 表、summary、既定 metric、
  完全共線、`p = 1`、learner failure。closed-form の learner は 1e-8 で一致。
- 非有限の予測値を learner failure として記録するようにした
  （scorer が黙って NaN を返す場合に `error` 列が空になる問題を修正）。
- rank deficiency の検証記録 `docs/rank_deficiency.md` を追加。

## 3. TODO

### 3.1 R 版との整合

- [x] R 参照値の再生成と parity test の整理（完了）。
- [ ] R 側が数値や結果表に影響する変更をしたら、参照値を再生成してテストを回す
      （手順は `CLAUDE.md` の "Regenerating the R reference"）。
- [ ] R 側にも同じ `inputs.json` を読むテストがあると双方向で検証できる（未実装）。

### 3.2 R 側への申し送り

- [x] すべて解決済み。rank deficiency に関する仕様書の記述2点は R の `cf2e1cc` で反映された
      （`docs/r_side_changes.md` に経緯を記録）。共通仕様に未解決項目はない。

### 3.3 リリースに向けて

- [ ] **PyPI 公開用 workflow** を作る（Docker とは独立した workflow。trusted publishing を想定）。
      README の `pip install sharedimp` は公開後に初めて動く。
- [ ] `pyproject.toml` / `CITATION.cff` の著者情報（所属、共著者、ORCID）を確定する。
      R 側には `CITATION.cff` が追加されたので内容を揃える。
- [ ] バージョン確定。現在 `0.1.0` のままで、2.2 以降の変更は CHANGELOG の Unreleased にある。
- [ ] 最初のリリースタグ `v0.1.0` を打つと Docker の version タグ（`0.1.0`, `0.1`）が公開される。
      タグ経由の公開はまだ一度も実行していない。
- [ ] SoftwareX 原稿に合わせた README・docs の最終確認。

### 3.4 検証の残り

- [ ] 公式 UMFI の分類パス（OOB accuracy）との数値比較は未実施。
- [ ] 公式 R パッケージ `HydroML/UMFI` は Python 側からは未実行
      （R 側の `docs/umfi_comparison.md` に比較記録あり）。
- [ ] 公開済み Docker イメージを手元で `docker pull` / `docker run` して確認していない
      （このマシンの開発アカウントには Docker デーモンへの権限がない）。
- [ ] 論文用の大規模 benchmark は `../sharedimp-experiments`（現在は空）で行う。

### 3.5 小さな後始末

- [ ] マージ済みブランチ `docker-publish` がローカルとリモートに残っている。
- [ ] `_direction.py`（未使用の experimental）を残すか削除するか。
      共通の direction 定義が決まるまでは公開しない。

## 4. 記録しておく差異と注意点

- **`split` の番号**: Python は 0 始まり、R は 1 始まり。比較は行の位置で行う。
- **rank 落ちした設計行列と OLS**: 互いに共線な列や training で定数の anchor がある場合、
  OLS の解は一意でなく、solver によって予測値が変わる（`docs/rank_deficiency.md`）。
  UMFI-LR の不一致ではなく下流 learner の問題。parity テストでは R の `lm.fit` の規約を
  再現するテスト専用 learner（`AliasingOLS`）を使う。ユーザーの estimator は置き換えない。
- **多クラスのロジスティック回帰**: R の `nnet::multinom` の収束精度のため、
  log loss の一致は約 3e-5 まで（テストの許容誤差は 1e-4）。
- **scikit-learn 1.3 のテスト補助関数**: ランナーの CPU 判定によって
  `check_estimators_pickle` が失敗することがあるため、その場合だけ skip している。
- **R リポジトリは並行して編集されている**。作業前に `git -C ../sharedimp-r status` で状態を確認する。
  R 側は Python の作業ツリー（editable install）を実行して確認することがあり、
  未コミットの変更が R 側の記録に入ることがある。
- **anaconda base 環境は使わない**（MKL が壊れていて NumPy の線形代数が落ちる）。
  検証は `.venv` と最小依存用の別 venv で行う。R は conda 環境
  `/home/koki/anaconda3/envs/sharedimp-dev/bin/Rscript`。
