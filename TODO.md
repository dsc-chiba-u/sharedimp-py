# sharedimp-py 作業記録と TODO

最終更新: 2026-11-13（`main` = `dd0bef0` 時点）

このファイルは作業の引き継ぎ用メモです。仕様そのものは
`../sharedimp-r/docs/specification.md`（R/Python 共通仕様の正本）、Python 側の適合状況は
`docs/specification.md`、R 側への申し送りは `docs/r_side_changes.md` を参照してください。

## 1. 現在の状態

| 項目 | 状態 |
| --- | --- |
| ブランチ | `main` = `dd0bef0`（push 済み、作業ツリーはクリーン） |
| pytest（scikit-learn 1.9.1） | 189 passed, 1 skipped（scikit-learn 側の array-API チェック） |
| pytest（最小依存: scikit-learn 1.3.2 / numpy 1.23.5 / pandas 1.5.3） | 178 passed |
| ruff / mypy / build / `twine check --strict` | すべて合格 |
| GitHub Actions `CI` | 成功（Python 3.10–3.13、macOS、Windows、最小依存、build） |
| GitHub Actions `DockerHub/GHCR` | 成功。`latest` と `dd0bef0` を公開済み |
| Docker イメージ | `ghcr.io/dsc-chiba-u/sharedimp-py`、`koki/sharedimp-py`（同一 digest） |
| PyPI | **未公開** |
| R 側で確認した版 | `sharedimp-r` `eaafaee`（読み取りのみ。R 側には一切書き込んでいない） |

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

## 3. TODO

### 3.1 R 側の完了待ち（指示があるまで着手しない）

- [ ] **R 参照値の再生成**。`tests/fixtures/cross_language/r_reference.json` は古い R コミット
      `6c5631d` の出力で、`tests/test_cross_language.py` が読み替え（列名での比較、
      raw metric の符号、行の並べ替え）をしている。R 側仕様改訂の完了が確認できたら:
  - [ ] `generate_r_reference.R` を新しい R の出力に合わせて直す
        （`score_*` 列、`dropped` → `collinear` フラグ、`drop_constant` 引数の廃止）。
  - [ ] 参照値を再生成し、テストの読み替えを外す。
  - [ ] 既定 metric どうしの比較と、失敗 fold（両言語 `on_error="warn"`）の比較を追加。
  - 手順は `CLAUDE.md` の "Regenerating the R reference" と `docs/r_side_changes.md` §C。
- [ ] 再生成後、`docs/specification.md` §5–§6 と `docs/r_side_changes.md` を更新。

### 3.2 R 側への申し送り（R リポジトリで対応してもらう）

実装の変更依頼はなし。仕様書の記述訂正のみ（詳細は `docs/r_side_changes.md` §B）。

- [ ] 既定 metric と `on_error` の実装コミットが `692c3fe` と書かれているが、正しくは `399bb20`。
- [ ] 「log loss のクリップは scikit-learn ≥ 1.5」→ 1.3.2 で既に machine epsilon。
- [ ] `accuracy` の対応: scikit-learn の `"accuracy"` は `predict` を使う。
      argmax の厳密な対応は `common_scorer("accuracy")`。
- [ ] §3 の「Python は `p = 1` を拒否する」は古い。
- [ ] §15.1 の共線バリアント: scikit-learn 1.3.2 の `LinearRegression` では一致しない
      （x3 が 45.55 でなく 41.79）。rank 判定を明示した OLS が必要である旨を追記。
- [ ] §16 の各項目の状態（仕様書ポインタ化は完了、ほかも commit 済み）。

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

- **非有限の予測値**: R は予測値を検査して失敗扱いにする。Python は予測値を直接見ないため、
  scorer が例外を出せば失敗として記録、出さなければ NaN の score として集計から外れる。
  集計値は同じで、`error` 列の有無だけが違う。
- **`split` の番号**: Python は 0 始まり、R は 1 始まり。比較は行の位置で行う。
- **共線な設計行列と OLS**: 互いに共線な列がある場合、古い scikit-learn の `LinearRegression`
  は R と違う値を返すことがある。parity テストでは rank 判定を明示した OLS を使っている。
- **scikit-learn 1.3 のテスト補助関数**: ランナーの CPU 判定によって
  `check_estimators_pickle` が失敗することがあるため、その場合だけ skip している。
- **R リポジトリは並行して編集されている**。作業前に `git -C ../sharedimp-r status` で状態を確認する。
  R 側は Python の作業ツリー（editable install）を実行して確認することがあり、
  未コミットの変更が R 側の記録に入ることがある。
- **anaconda base 環境は使わない**（MKL が壊れていて NumPy の線形代数が落ちる）。
  検証は `.venv` と最小依存用の別 venv で行う。R は conda 環境
  `/home/koki/anaconda3/envs/sharedimp-dev/bin/Rscript`。
