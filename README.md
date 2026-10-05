# 水匠BM（Suisho-BM）

将棋の必至問題を解く USI エンジンです。
長井歩「難解な必至問題を解くアルゴリズムとその実装」（GPW 2011）の方式（仮想パスと詰みフラグによる単一の df-pn+）を、
[やねうら王](https://github.com/yaneurao/YaneuraOu) の USER_ENGINE として実装しています。

- 必至を見つけると、その証明を別の処理で独立に検証してから答えます（受方の全合法手と仮想パスを確認し、循環も調べます）。
- 解答手順は、受方は最長、攻方は最短の手順です。攻方の最短は、見つけた証明の範囲での最短です（予算付きの追加探索で短縮します）。`HisshiFutile=1` で無駄合いを除きます（連続合駒も末尾から判定します）。
- 複数スレッドで置換表を共有して探索します。検証と解答作成も並列に行います。
- 探索フェーズごとの時間配分やノード配分はありません。一つの df-pn+ 探索を、解けるか止められるまで続けます。

## 探索の概要

- 受方の手番で王手がかかっていない局面には「仮想パス」を加えます。パスの後は攻方の王手だけを生成するので（詰みモード）、パスの子が証明されれば直前の攻めは詰めろです。
- 攻方の非王手は論文 3.1 節の候補手に絞り、証明数に対するコストを 2 とします（df-pn+）。受方は全合法手を生成します。
- 同じ方向からの攻めや同じ駒を取る受けは、最大値で一つのグループとして数えます（論文 3.1 節）。
- 証明駒（論文 3.3 節）と、手駒の優越による置換表の参照を使います。
- 指し手は段階的・遅延的に生成します。王手、パス、関連する受け、その他の受けの順で、中合いは最初、1 マスにつき 1 種類だけ生成します。
- パスの詰み手順を他の受けに再生して証明を試みます（証明の再生）。
- 未知の局面の初期証明数を深さに応じて増やし（Deep df-pn）、1 スレッドのときは 1+ε 法を使います。
- 置換表のエントリは 24 バイトです。置換は KomoringHeights と同様に探索量を基準にし、証明・反証は重く扱います。
- 補助スレッドは子の順序と探索パラメータ（非王手コスト、遅延生成の推定値、ε、深さ係数）をスレッドごとに変えて、探索を多様にします。

## 配布ファイル（Windows）

[Releases](https://github.com/tayayan/Suisho-BM/releases) の zip に、CPU 別の実行ファイルが入っています。

| ファイル | 対象 CPU |
|---|---|
| `Suisho-BM-AVX2.exe` | AVX2 と BMI2 に対応した CPU（Intel Haswell 以降、AMD Zen 3 以降） |
| `Suisho-BM-ZEN2.exe` | AMD Zen / Zen+ / Zen 2（Ryzen 1000〜3000 番台など。BMI2 が遅い CPU） |
| `Suisho-BM-SSE41.exe` | 上記が動かない古い CPU |

どれも探索結果は同じで、違うのは速さだけです。

## ビルド

Windows では [MSYS2](https://www.msys2.org/) の MinGW64 環境（g++）でビルドします。

```powershell
.\build.ps1                     # AVX2（Intel 等）
.\build.ps1 -TargetCpu ZEN2     # AMD Zen 2（BMI2 が遅い CPU）
.\build.ps1 -Rebuild            # 全オブジェクトを作り直す
```

MSYS2 の既定のインストール先は `C:\Tools\msys2` です。別の場所に入れた場合は `-Msys2Root` で指定してください。
MSYS2 のシェルから直接 make を実行することもできます（Linux でも同様にビルドできます）。

```bash
cd source
make -j8 TARGET_CPU=AVX2
```

`TARGET_CPU` には `AVX2`、`ZEN2`、`ZEN3`、`AVX512`、`SSE42`、`SSE41`、`SSE2` を指定できます。
実行ファイルはリポジトリ直下の `Suisho-BM.exe` です（Linux では `Suisho-BM`）。

## 使い方（USI）

USI プロトコルで動きます。ShogiGUI などの GUI にエンジンとして登録し、詰将棋（詰み探索）の機能で問題局面を解かせてください。

### コマンド例

```
usi
setoption name USI_Hash value 2048
setoption name Threads value 8
isready
position sfen 7nl/7k1/6p2/7Pp/9/9/9/9/9 b GS2r2b3g3s3n3l15p 1
go mate infinite
```

この例では、検証済みの解答が次のように返ります。

```
checkmate S*2c 2b1c 2c3d+
```

### 探索コマンド

| コマンド | 動作 |
|---|---|
| `go mate infinite` / `go mate <ms>` | 必至を探索し、`checkmate <手順>`、`checkmate nomate`（不必至）、`checkmate timeout`（時間切れ・停止）のいずれかを返します |
| `go infinite` / `go movetime <ms>` / `go nodes <n>` | 同じ探索の後、`bestmove` で初手を返します（必至がなければ `bestmove resign`） |

`stop` で探索を止められます。手番の側が攻方です。

### オプション

| オプション | 既定値 | 内容 |
|---|---|---|
| `USI_Hash` | 1024 | 置換表の大きさ（MB）。難しい問題では 2048MB 程度に増やすと安定します。置換表のほかに、この値の 4 割程度までの補助メモリ（子局面のキャッシュ・検証用の表）を使います |
| `Threads` | 1 | 探索スレッド数。検証と解答作成も、このスレッド数で並列に行います |
| `PvInterval` | 1000 | 進捗出力の間隔（ms）。0 にすると出力しません |
| `HisshiFullWidth` | false | 攻方も全合法手を生成します（遅くなります）。不必至の結論が、全ての攻めについてのものになります |
| `HisshiFutile` | 0 | 解答手順の無駄合いの扱い。0 は合駒もすべて受けとして手順に含めます。1 は無駄合いを除きます（解答作成に時間がかかり、手数制限付き探索のために置換表とは別に 128MB を使います） |

### 出力

- 探索中は `info depth ... nodes ... nps ... hashfull ... score cp ... pv ...` を定期的に出力します。score は根の pn/dn から求めた評価値（-600·log(pn/dn)）で、pv は現在の最善手順です。pn/dn の値そのものは `info string pn=... dn=...` で出力します。
- 証明が見つかると、検証中、解答作成中であることを `info string` で知らせます。
- 解けると `info ... score mate +<手数> pv <手順>` と `info string proof_verified=1 ...` を出力します。

### 結果の読み方

- 必至が成立するのは、`proof_verified=1` と手順付きの `checkmate ...` が出たときです。
- 解答手順は、受方は最長、攻方は最短の手順です（`HisshiFutile=1` では無駄合いを除きます）。手順の最後は必至をかける攻方の手です。
- 必至の成立は独立に検証済みですが、攻方の最短は見つけた証明の範囲での最短で、真の最短は保証しません。予算付きの追加探索で短い手順を探しますが、探索の進み方によって手数が変わることがあります。
- `checkmate timeout` は未解決という意味で、不必至と判定したわけではありません。
- `go mate <ms>` や `go nodes <n>` の制限は、探索・検証・解答作成の全体（全スレッドの合計）に適用されます。解答作成の途中で制限に達したときは、それまでに分かった範囲の手順（`answer=greedy`）を返します。検証の途中で制限に達したときは `checkmate timeout` になり、`proof_verified=0 ... stopped=limit` と出力します。
- 既定の候補手モードの `checkmate nomate` は、候補手の範囲で必至がないという意味です。全ての攻めについての結論には `HisshiFullWidth=true` を使ってください。
- 複数スレッドでは、スレッド間のタイミングによって探索時間と解答手順が実行ごとに変わることがあります（どの手順も検証済みです）。1 スレッドの結果は毎回同じです。

## ファイル構成

| パス | 内容 |
|---|---|
| `source/engine/user-engine/hisshi_dfpn.hpp` | 探索の設定・結果・置換表（外部とのインターフェース） |
| `source/engine/user-engine/search.hpp` | 必至探索（仮想パス付き df-pn+、指し手生成、証明の再生） |
| `source/engine/user-engine/verify.hpp` | 証明・反証の独立検証 |
| `source/engine/user-engine/answer.hpp` | 解答手順の作成（最長の受け・最短の攻め・無駄合いの除外） |
| `source/engine/user-engine/solver.cpp` | 置換表の管理と全体の流れ（並列探索 → 検証 → 解答作成） |
| `source/engine/user-engine/user-search.cpp` | USI との接続 |
| `source/` のその他 | やねうら王から必要な部分だけを抜き出したもの（盤面・指し手生成・1 手詰め・USI など） |
| `build.ps1` / `source/Makefile` | ビルド |

## クレジット・ライセンス

- 長井歩「難解な必至問題を解くアルゴリズムとその実装」第16回ゲームプログラミングワークショップ（GPW 2011）
- [やねうら王](https://github.com/yaneurao/YaneuraOu)：盤面表現、指し手生成、1 手詰め判定、USI の土台
- [KomoringHeights](https://github.com/komori-n/KomoringHeights)：置換表の置換方針や進捗出力の形式を参考にしました

やねうら王・KomoringHeights と同じく、GNU General Public License v3.0 で公開します（[LICENSE](LICENSE)）。
