<img src="https://github.com/generative-computing/mellea/raw/main/docs/mellea_draft_logo_300.png" height=100>

# Mellea

> **注記:** この文書は英語版から翻訳されたものです。最新の更新については、オリジナルのファイルを参照してください。

Melleaは、生成的プログラムを記述するためのライブラリです。
生成的プログラミングは、不安定なエージェントや脆弱なプロンプトを、
構造化された、保守可能で、堅牢かつ効率的なAIワークフローに置き換えます。


[//]: # ([![arXiv]&#40;https://img.shields.io/badge/arXiv-2408.09869-b31b1b.svg&#41;]&#40;https://arxiv.org/abs/2408.09869&#41;)
[![Docs](https://img.shields.io/badge/docs-live-brightgreen)](https://mellea.ai/)
[![PyPI version](https://img.shields.io/pypi/v/mellea)](https://pypi.org/project/mellea/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/mellea)](https://pypi.org/project/mellea/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![GitHub License](https://img.shields.io/github/license/generative-computing/mellea)](https://img.shields.io/github/license/generative-computing/mellea)
[![Discord](https://img.shields.io/discord/1448407063813165219?logo=discord&logoColor=white&label=Discord&color=7289DA)](https://ibm.biz/mellea-discord)


## 機能

 * 独自のプロンプティングパターンの標準ライブラリ
 * 推論時スケーリングのためのサンプリング戦略
 * 検証器とサンプラー間のクリーンな統合
    - バッテリー同梱の検証器ライブラリ
    - 活性化されたLoRAを使用した特殊要件の効率的なチェックのサポート
    - 独自の分類器データで独自の検証器をトレーニング可能
 * 多くの推論サービスやモデルファミリーと互換性があります。以下の間でワークロードを簡単に移行することで、コストと品質を制御できます：
        - 推論プロバイダー
        - モデルファミリー
        - モデルサイズ
 * レガシーコードベースにLLMのパワーを簡単に統合（mify）
 * 仕様を記述し、`mellea`に詳細を埋めさせることでアプリケーションをスケッチ（generative slots）
 * 大きくて扱いにくいプロンプトを構造化された保守可能なmelleaの問題に分解することから始められます



## はじめに

ローカルインストールまたはColabノートブックを使用して始めることができます。

### ローカル推論で始める

<img src="https://github.com/generative-computing/mellea/raw/main/docs/GetStarted_py.png" style="max-width:800px">

[uv](https://docs.astral.sh/uv/getting-started/installation/)でインストール：

```bash
uv pip install mellea
```

pipでインストール：

```bash
pip install mellea
```

> [!NOTE]
> `mellea`には、`pyproject.toml`で定義されている追加パッケージが付属しています。すべてのオプションの依存関係をインストールする場合は、以下のコマンドを実行してください：
>
> ```bash
> uv pip install "mellea[hf]" # Huggingfaceエクストラとアロラ機能用
> uv pip install "mellea[watsonx]" # watsonxバックエンド用
> uv pip install "mellea[docling]" # docling用
> uv pip install "mellea[all]" # すべてのオプション依存関係用
> ```
>
> `uv sync --all-extras`ですべてのオプション依存関係をインストールすることもできます

> [!NOTE]
> Intel Macで実行している場合、torch/torchvisionのバージョンに関連するエラーが発生する可能性があります。Condaはこれらのパッケージの更新版を維持しています。conda環境を作成し、`conda install 'torchvision>=0.22.0'`を実行する必要があります（これによりpytorchとtorchvision-extraもインストールされます）。その後、`uv pip install mellea`を実行できるはずです。サンプルを実行するには、`uv run --with mellea <filename>`の代わりに、conda環境内で`python <filename>`を使用する必要があります。

> [!NOTE]
> python >= 3.13を使用している場合、rustコンパイラの問題（`error: can't find Rust compiler`）によりoutlinesがインストールできない問題が発生する可能性があります。python 3.12にダウングレードするか、[rustコンパイラ](https://www.rust-lang.org/tools/install)をインストールしてoutlinesのwheelをローカルでビルドすることができます。

ローカルで簡単なLLMリクエストを実行するには（OllamaとGraniteモデルを使用）、以下のコードから始めます：
```python
# file: https://github.com/generative-computing/mellea/blob/main/docs/examples/tutorial/example.py
import mellea

m = mellea.start_session()
print(m.chat("What is the etymology of mellea?").content)
```


次に実行します：
> [!NOTE]
> 始める前に、[ollama](https://ollama.com/)をダウンロードしてインストールする必要があります。Melleaは多くの異なるタイプのバックエンドで動作しますが、このチュートリアルのすべては、IBMのGranite 4 Micro 3Bモデルを実行しているMacbookで「そのまま動作」します。
```shell
uv run --with mellea docs/examples/tutorial/example.py
```

### Colabで始める

| ノートブック | Colabで試す | 目的 |
|----------|--------------|------|
| Hello, World | <a target="_blank" rel="noopener noreferrer" href="https://colab.research.google.com/github/generative-computing/mellea/blob/main/docs/examples/notebooks/example.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> | クイックスタートデモ |
| Simple Email | <a target="_blank" rel="noopener noreferrer" href="https://colab.research.google.com/github/generative-computing/mellea/blob/main/docs/examples/notebooks/simple_email.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> | `m.instruct`プリミティブの使用 |
| Instruct-Validate-Repair | <a target="_blank" rel="noopener noreferrer" href="https://colab.research.google.com/github/generative-computing/mellea/blob/main/docs/examples/notebooks/instruct_validate_repair.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> | 最初の生成的プログラミングデザインパターンの紹介 |
| Model Options | <a target="_blank" rel="noopener noreferrer" href="https://colab.research.google.com/github/generative-computing/mellea/blob/main/docs/examples/notebooks/model_options_example.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> | バックエンドにモデルオプションを渡す方法のデモ |
| Sentiment Classifier | <a target="_blank" rel="noopener noreferrer" href="https://colab.research.google.com/github/generative-computing/mellea/blob/main/docs/examples/notebooks/sentiment_classifier.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> | `@generative`デコレータの紹介 |
| Managing Context | <a target="_blank" rel="noopener noreferrer" href="https://colab.research.google.com/github/generative-computing/mellea/blob/main//docs/examples/notebooks/context_example.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> | `MelleaSession`でコンテキストを構築・管理する方法 |
| Generative OOP | <a target="_blank" rel="noopener noreferrer" href="https://colab.research.google.com/github/generative-computing/mellea/blob/main/docs/examples/notebooks/table_mobject.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> | Melleaでのオブジェクト指向生成的プログラミングのデモ |
| Rich Documents | <a target="_blank" rel="noopener noreferrer" href="https://colab.research.google.com/github/generative-computing/mellea/blob/main/docs/examples/notebooks/document_mobject.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> | Doclingを使用してリッチテキストドキュメントを扱う生成的プログラム |
| Composing Generative Functions | <a target="_blank" rel="noopener noreferrer" href="https://colab.research.google.com/github/generative-computing/mellea/blob/main/docs/examples/notebooks/compositionality_with_generative_slots.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> | Melleaでの契約指向プログラミングのデモ |
| `m serve` | <a target="_blank" rel="noopener noreferrer" href="https://colab.research.google.com/github/generative-computing/mellea/blob/main/docs/examples/notebooks/m_serve_example.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> | 生成的プログラムをopenai互換のモデルエンドポイントとして提供 |
| MCP | <a target="_blank" rel="noopener noreferrer" href="https://colab.research.google.com/github/generative-computing/mellea/blob/main/docs/examples/notebooks/mcp_example.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> | Mellea + MCP |


### ソースからの`uv`ベースのインストール

リポジトリをフォークしてクローン：

```bash
git clone ssh://git@github.com/<my-username>/mellea.git && cd mellea/
```

仮想環境をセットアップ：

```bash
uv venv .venv && source .venv/bin/activate
```

`uv pip`を使用して編集可能フラグ付きでソースからインストール：

```bash
uv pip install -e ".[all]"
```

リポジトリに貢献する予定がある場合は、すべての開発要件をインストールすることをお勧めします：

```bash
uv pip install ".[all]" --group dev --group notebook --group docs
```

または

```bash
uv sync --all-extras --all-groups
```

貢献したい場合は、precommitフックをインストールしてください：

```bash
pre-commit install
```

### ソースからの`conda`/`mamba`ベースのインストール

リポジトリをフォークしてクローン：

```bash
git clone ssh://git@github.com/<my-username>/mellea.git && cd mellea/
```

上記のすべてのコマンドを実行するインストールスクリプトが付属しています：

```bash
conda/install.sh
```

## 検証を使った始め方

Melleaは、**instruct-validate-repair**パターンを通じて生成結果の検証をサポートしています。
以下では、*"Write an email.."*のリクエストが*"be formal"*と*"Use 'Dear interns' as greeting."*の要件によって制約されています。
シンプルな棄却サンプリング戦略を使用して、リクエストは最大3回（loop_budget）モデルに送信され、
出力は（この場合）LLM-as-a-judgeを使用して制約に対してチェックされます。


```python
# file: https://github.com/generative-computing/mellea/blob/main/docs/examples/instruct_validate_repair/101_email_with_validate.py
from mellea import MelleaSession
from mellea.backends import ModelOption
from mellea.backends.ollama import OllamaModelBackend
from mellea.backends import model_ids
from mellea.stdlib.sampling import RejectionSamplingStrategy

# Ollama上で動作するMistralでセッションを作成
m = MelleaSession(
    backend=OllamaModelBackend(
        model_id=model_ids.MISTRALAI_MISTRAL_0_3_7B,
        model_options={ModelOption.MAX_NEW_TOKENS: 300},
    )
)

# 要件付きで指示を実行
email_v1 = m.instruct(
    "Write an email to invite all interns to the office party.",
    requirements=["be formal", "Use 'Dear interns' as greeting."],
    strategy=RejectionSamplingStrategy(loop_budget=3),
)

# 結果を出力
print(f"***** email ****\n{str(email_v1)}\n*******")
```


## Generative Slotsで始める

Generative slotsを使用すると、実装せずに関数を定義できます。
`@generative`デコレータは、LLMにクエリすることで解釈されるべき関数をマークします。
以下の例は、LLMの感情分類機能を、Melleaのgenerative slotsとローカルLLMを使用して
関数としてラップする方法を示しています。


```python
# file: https://github.com/generative-computing/mellea/blob/main/docs/examples/tutorial/sentiment_classifier.py#L1-L13
from typing import Literal
from mellea import generative, start_session


@generative
def classify_sentiment(text: str) -> Literal["positive", "negative"]:
  """Classify the sentiment of the input text as 'positive' or 'negative'."""


if __name__ == "__main__":
  m = start_session()
  sentiment = classify_sentiment(m, text="I love this!")
  print("Output sentiment is:", sentiment)
```



## チュートリアル

[チュートリアル](docs/tutorial.md)を参照してください

## Melleaへの貢献

すべてのMelleaコードがこのリポジトリにあるわけではありません。
Melleaへの貢献には3つの経路があります：
1. アプリケーション、ツール、ライブラリの貢献。これらは独自のリポジトリでホストできます。
   可視性のために、`mellea-`プレフィックスを使用してください。例：
   `github.com/my-company/mellea-legal-utils`または`github.com/my-username/mellea-swe-agent`。
2. スタンドアロンで汎用的なComponents、Requirements、またはSampling Strategiesの貢献。
   提案する機能を説明する**issueを開き**、貢献が標準ライブラリ（このリポジトリ）に
   入るべきか、[mellea-contribs](https://github.com/generative-computing/mellea-contribs)
   ライブラリに入るべきかについて、コアチームからフィードバックを得てください。
   issueがトリアージされた後、関連するリポジトリでPRを開いてください。
3. Melleaコアへの新機能の貢献、またはMelleaコアや標準ライブラリのバグ修正。
   バグまたは機能を説明する**issueを開いて**ください。issueがトリアージされた後、
   このリポジトリでPRを開き、自動化されたPRワークフローの指示に従ってください。

### このリポジトリへの貢献

Melleaに貢献する場合は、precommitフックを使用することが重要です。
これらのフックを使用する、またはテストスイートを実行するには、
`[all]`オプション依存関係とdevグループをインストールする必要があります。

```
git clone git@github.com:generative-computing/mellea.git && 
cd mellea && 
uv venv .venv && 
source .venv/bin/activate &&
uv pip install -e ".[all]" --group dev
pre-commit install
```

その後、`pytest`を実行してすべてのテストを実行するか、
`CICD=1 pytest`を実行してCI/CDテストのみを実行できます。
特定のテストカテゴリ（例：バックエンド別、リソース要件別）の実行については、
[test/MARKERS_GUIDE.md](test/MARKERS_GUIDE.md)を参照してください。

ヒント：`git commit`に`-n`フラグを渡すことでフックをバイパスできます。
これは、後でスカッシュする予定の中間コミットに役立つことがあります。

貢献方法の詳細な追加手順については、[Contributor Guide](docs/tutorial.md#appendix-contributing-to-mellea)を参照してください。

### IBM ❤️ Open Source AI

Melleaは、マサチューセッツ州ケンブリッジのIBM Researchによって開始されました。