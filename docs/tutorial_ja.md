# 生成的プログラミングの原則：Melleaアプローチ

> **注記:** この文書は英語版から翻訳されたものです。最新の更新については、オリジナルのファイルを参照してください。

## 目次

- [生成的プログラミングの原則：Melleaアプローチ](#生成的プログラミングの原則melleaアプローチ)
  - [目次](#目次)
  - [第1章：生成的プログラミングとは](#第1章生成的プログラミングとは)
  - [第2章：Melleaで生成的プログラミングを始める](#第2章melleaで生成的プログラミングを始める)
    - [Requirements（要件）](#requirements要件)
    - [要件の検証](#要件の検証)
    - [Instruct - Validate - Repair](#instruct---validate---repair)
    - [ModelOptions](#modeloptions)
      - [システムメッセージ](#システムメッセージ)
    - [まとめ](#まとめ)
  - [第3章：標準ライブラリの概要](#第3章標準ライブラリの概要)
  - [第4章：Generative Slots](#第4章generative-slots)
      - [例：感情分類器](#例感情分類器)
      - [Generative Slotsを使用してモジュール境界を越えた構成可能性を提供する](#generative-slotsを使用してモジュール境界を越えた構成可能性を提供する)
  - [第5章：MObjects](#第5章mobjects)
    - [例：MObjectとしてのテーブル](#例mobjectとしてのテーブル)
    - [ケーススタディ：ドキュメントの操作](#ケーススタディドキュメントの操作)
    - [MObjectメソッドはツールです](#mobjectメソッドはツールです)
  - [第6章：RequirementsとComponentsのチューニング](#第6章requirementsとcomponentsのチューニング)
    - [問題の定義](#問題の定義)
    - [aLoRAアダプターのトレーニング](#aloraアダプターのトレーニング)
      - [パラメータ](#パラメータ)
    - [Hugging Faceへのアップロード（オプション）](#hugging-faceへのアップロードオプション)
    - [チューニングされたモデルのMelleaへの統合](#チューニングされたモデルのmelleaへの統合)
  - [第7章：コンテキスト管理について](#第7章コンテキスト管理について)
  - [第8章：エージェントの実装](#第8章エージェントの実装)
    - [ケーススタディ：MelleaでReACTを実装する](#ケーススタディmelleaでreactを実装する)
    - [保護された非決定性](#保護された非決定性)
  - [第9章：他のフレームワークとの相互運用性](#第9章他のフレームワークとの相互運用性)
    - [Melleaを実行するシンプルなmcpサーバー](#melleaを実行するシンプルなmcpサーバー)
    - [MelleaプログラムをOpenAI互換サーバーとして実行する（実験的）](#melleaプログラムをopenai互換サーバーとして実行する実験的)
      - [例：`m serve`アプリケーション](#例m-serveアプリケーション)
  - [第10章：Mのためのプロンプトエンジニアリング](#第10章mのためのプロンプトエンジニアリング)
    - [テンプレート](#テンプレート)
    - [テンプレート表現](#テンプレート表現)
    - [カスタマイズ](#カスタマイズ)
      - [テンプレートの選択](#テンプレートの選択)
      - [既存のクラスの編集](#既存のクラスの編集)
  - [第11章：ツール呼び出し](#第11章ツール呼び出し)
  - [第12章：非同期性](#第12章非同期性)
    - [非同期関数](#非同期関数)
    - [同期関数における非同期性](#同期関数における非同期性)
  - [付録：Melleaへの貢献](#付録melleaへの貢献)
    - [コントリビューターガイド：はじめに](#コントリビューターガイドはじめに)
    - [コントリビューターガイド：RequirementsとVerifiers](#コントリビューターガイドrequirementsとverifiers)
    - [コントリビューターガイド：Components](#コントリビューターガイドcomponents)

## 第1章：生成的プログラミングとは

このチュートリアルはMelleaについてです。Melleaは、より良い生成的プログラムを書くのに役立ちます。

*生成的プログラム*とは、LLMへの呼び出しを含むあらゆるコンピュータプログラムです。このチュートリアル全体を通して見ていくように、LLMはさまざまな方法でソフトウェアに組み込むことができます。LLMをプログラムに組み込む方法の中には、堅牢でパフォーマンスの高いシステムを生み出す傾向があるものもあれば、脆弱でエラーが発生しやすいソフトウェアを生み出すものもあります。

生成的プログラムは、生成モデルを呼び出す関数の使用によって古典的なプログラムと区別されます。これらの生成的呼び出しは、文字列、ブール値、構造化データ、コード、画像/動画など、多くの異なるデータ型を生成できます。生成的呼び出しの基礎となるモデルとソフトウェアは、特定の状況や特定の方法で組み合わせたり構成したりできます（例：特殊なケースとしてのLoRAアダプター）。生成的呼び出しを呼び出すことに加えて、生成的プログラムは、ベースにLLMを持たない言語で書かれた他の関数を呼び出すこともできます。これにより、例えば、生成的関数の出力をDBリトリーバルシステムに渡し、その出力を別のジェネレーターに供給することができます。生成的プログラムを書くことが難しいのは、生成的プログラムが決定論的操作と確率的操作を交互に実行するためです。

要件検証は、生成的プログラムにおける非決定性の期間を限定する上で重要な役割を果たします。ブール値や他の出力を生成する検証器を実装し、検証器がyesと言うまで、あるいは反復回数が多くなりすぎて例外処理プロセスをトリガーするまでループを繰り返すことができます。したがって、生成的関数の出力における確実性の度合いを決定し、その確実性の量に基づいて行動することができます。検証は、生成的関数へのクエリから正確なプログラム的チェック、そしてそれらのさまざまな組み合わせまで、さまざまな方法で行うことができます。

長い計算パスを含むプログラム（反復や再帰を含むほとんどのプログラム）では、不確実性の増分的な蓄積は乗法的であり、したがって、生成的プログラムの実行全体を通じて増分的な要件検証によって時折限定される必要があります。これらの増分的なチェックは、変動のパターンや不変のプロパティを確立するために使用でき、どちらも実行が望ましい状態に収束し、「間違った方向に進まない」ことを保証するのに役立ちます。これらの増分的なチェックの構築は、生成的プログラミングにおける重要なタスクの1つであり、それ自体が生成的プログラミングに適したタスクとして扱うことができます。他の要件チェックと同様に、これらのバリアントと不変量は明示的でプログラム的であるか、生成的関数を介して解決できます。いずれにせよ、各生成的プログラムは計算のトレースを生成します - 成功したものもあれば、失敗したものもあります。

失敗パスについて何をすべきかを考えることは、生成的プログラムの作成者が直面するもう1つの重要な問題です。成功したトレースを収集して、最終的に高信頼度の結果を得ることができます。あるいは、いくつかの失敗や低信頼度の回答を含むトレースが蓄積される可能性があります。生成的プログラムは、これらの失敗した検証を修復しようとします。修復プロセスは手動、自動化、またはユーザーインタラクションと自動化された修復メカニズムの組み合わせを提供できます。生成的プログラムがこのように実行されると、コンテキストが蓄積されます。ますます大きくなるコンテキストの蓄積は、それ自体が課題となります。

したがって、メモリ管理はコンテキストエンジニアリングにおいて重要な役割を果たします。したがって、Melleaは、KV Cacheのコンポーネントを開発者およびユーザー向けの抽象化にマッピングし、コンテキストの構築とキャッシュされたキーと値の処理を自動化するメカニズムを提供します。

Mellea開発者がこの生成的プログラミングのライブラリを構築する際、このチュートリアル全体を通して繰り返し現れる有用な原則をいくつか見つけました：

 * **LLM呼び出しを要件検証器で限定する。** この原則のバリエーションをチュートリアル全体で見ていきます。
 * **生成的プログラムはシンプルで構成可能なプロンプティングスタイルを使用すべきです。** Melleaは、「フレームワークがプロンプトを選択する」パラダイムと「クライアントコードがプロンプトを選択する」パラダイムの中間を取ります。プロンプトを小さく自己完結的に保ち、多くのそのようなプロンプトを連鎖させることで、通常、いくつかのプロンプトスタイルの1つで済ませることができます。新しいプロンプトスタイルが必要な場合、そのプロンプトはそれを使用するソフトウェアと共同設計されるべきです。Melleaでは、生成的プログラムを*Components*に分解することでこれを奨励しています。詳細は[第3章](#第3章標準ライブラリの概要)で説明します。
 * **生成モデルと推論時プログラムは共同設計されるべきです。** 理想的には、推論時に使用されるプロンプティングのスタイルとドメインは、事前トレーニング、中間トレーニング、および/またはポストトレーニングで使用されるプロンプティングのスタイルとドメインと一致する必要があります。同様に、モデルはランタイムコンポーネントと使用パターンを念頭に置いて構築されるべきです。第6章でこの初期の例をいくつか見ていきます。
 * **生成的プログラムはコンテキストを慎重に管理すべきです。** 各Componentは単一の呼び出しのコンテキストを管理します。これは第2章、第3章、第4章、第5章で見ていきます。さらに、Melleaは複数の呼び出しにわたってコンテキストを再利用するための有用なメカニズムを提供します（第7章）。

優れた生成的プログラムはどの言語やフレームワークでも書くことができますが、正しく行うことは簡単ではありません。MelleaはLLMライブラリの設計空間における1つのポイントに過ぎませんが、私たちはそれが良いものだと考えています。私たちの希望は、Melleaが堅牢で、パフォーマンスが高く、目的に適した生成的プログラムを書くのに役立つことです。

## 第2章：Melleaで生成的プログラミングを始める

始める前に、[ollama](https://ollama.com/)をダウンロードしてインストールする必要があります。Melleaは多くの異なるタイプのバックエンドで動作しますが、このチュートリアルのすべては、IBMのGranite 4 Micro 3Bモデルを実行しているMacbookで「そのまま動作」します。

また、[uv](https://docs.astral.sh/uv/#installation)をダウンロードしてインストールすることをお勧めします。チュートリアルの例は次のように実行できます：
```bash
uv run example_name.py --with mellea
```
> [!NOTE]
> Intel Macで実行している場合、torch/torchvisionのバージョンに関連するエラーが発生する可能性があります。Condaはこれらのパッケージの更新版を維持しています。conda環境を作成し、`conda install 'torchvision>=0.22.0'`を実行する必要があります（これによりpytorchとtorchvision-extraもインストールされます）。その後、`uv pip install mellea`を実行できるはずです。サンプルを実行するには、`uv run --with mellea <filename>`の代わりに、conda環境内で`python <filename>`を使用する必要があります。

> [!NOTE]
> python >= 3.13を使用している場合、rustコンパイラの問題（`error: can't find Rust compiler`）によりoutlinesがインストールできない問題が発生する可能性があります。python 3.12にダウングレードするか、[rustコンパイラ](https://www.rust-lang.org/tools/install)をインストールしてoutlinesのwheelをローカルでビルドすることができます。

ollamaをインストールして実行したら、最初の生成的コードを始めることができます：

```python
# file: https://github.com/generative-computing/mellea/blob/main/docs/examples/tutorial/simple_email.py#L1-L8
import mellea

# INFO: この行はIBMのGranite 4 Micro 3Bモデルをダウンロードします。
m = mellea.start_session()

email = m.instruct("Write an email inviting interns to an office party at 3:30pm.")
print(str(email))
```

ここでは、granite3.3-chatモデルを使用してローカルマシン上でOllamaを実行するバックエンドを初期化しました。
次に、モデルにメールを生成するように依頼し、それをコンソールに出力します。

> [!NOTE]
> Melleaは他の多くのモデルとバックエンドをサポートしています。デフォルトでは、新しいMelleaセッションは、自分のラップトップ上でIBMの有能なGranite 3Bモデルを実行します。これは始めるのに良い（そして無料の！）方法です。他のモデルやバックエンドを試したい場合は、start_sessionメソッドでバックエンドとモデルを明示的に指定できます。例：`mellea.start_session(backend_name="ollama", model_id=mellea.model_ids.IBM_GRANITE_4_MICRO_3B)`。

続ける前に、この呼び出しをいくつかの引数を持つ関数にラップしましょう：

```python
# file: https://github.com/generative-computing/mellea/blob/main/docs/examples/tutorial/simple_email.py#L13-L27
import mellea

def write_email(m: mellea.MelleaSession, name: str, notes: str) -> str:
  email = m.instruct(
    "Write an email to {{name}} using the notes following: {{notes}}.",
    user_variables={"name": name, "notes": notes},
  )
  return email.value  # str(email)も機能します。

m = mellea.start_session()
print(write_email(m, "Olivia",
                  "Olivia helped the lab over the last few weeks by organizing intern events, advertising the speaker series, and handling issues with snack delivery."))
```

これで、メール作成関数ができました！

instructメソッドが`user_variables`として変数の辞書を受け取ることができることに注目してください。これらは、指示の説明をjinjaテンプレートとして扱うことで埋め込まれます。

`m.instruct()`関数はデフォルトで`ModelOutputThunk`を返し、モデル出力文字列がフィールド`.value`にバインドされています。

### Requirements（要件）

しかし、生成されたメールが良いものであることをどのように知るのでしょうか？
優れた生成的プログラマーはこれを偶然に任せません - 代わりに、LLMへの入力が期待通りであることを保証するために事前条件を使用し、LLMの出力が目的に適していることを保証するために事後条件をチェックします。

この場合、メールに挨拶があり、小文字のみが含まれていることを確認したいとします。これらの事後条件を、`m.instruct`呼び出しに**requirements**を指定することでキャプチャできます：

```python
# file: https://github.com/generative-computing/mellea/blob/main/docs/examples/tutorial/simple_email.py#L33-L53
import mellea

def write_email_with_requirements(m: mellea.MelleaSession, name: str, notes: str) -> str:
  email = m.instruct(
      "Write an email to {{name}} using the notes following: {{notes}}.",
      requirements=[
          "The email should have a salutation",
          "Use only lower-case letters",
      ],
      user_variables={"name": name, "notes": notes},
  )
  return str(email)

m = mellea.start_session()
print(write_email_with_requirements(
  m,
  name="Olivia",
  notes="Olivia helped the lab over the last few weeks by organizing intern events, advertising the speaker series, and handling issues with snack delivery.",
))
```

モデルリクエストに追加される2つの要件を指示に追加しました。しかし、これらの要件が満たされているかどうかはまだチェックしていません。要件を検証するための**strategy**を追加しましょう：

```python
# file: https://github.com/generative-computing/mellea/blob/main/docs/examples/tutorial/simple_email.py#L57-L84
import mellea
from mellea.stdlib.sampling import RejectionSamplingStrategy

def write_email_with_strategy(m: mellea.MelleaSession, name: str, notes: str) -> str:
    email_candidate = m.instruct(
        "Write an email to {{name}} using the notes following: {{notes}}.",
        requirements=[
            "The email should have a salutation",
            "Use only lower-case letters",
        ],
        strategy=RejectionSamplingStrategy(loop_budget=5),
        user_variables={"name": name, "notes": notes},
        return_sampling_results=True,
    )
    if email_candidate.success:
        return str(email_candidate.result)
    else:
        print("Expect sub-par result.")
        return email_candidate.sample_generations[0].value

m = mellea.start_session()
print(
    write_email_with_strategy(
        m,
        "Olivia",
        "Olivia helped the lab over the last few weeks by organizing intern events, advertising the speaker series, and handling issues with snack delivery.",
    )
)
```

ここでいくつかのことが起こりました。まず、指示にサンプリング`strategy`を追加しました。
この戦略（`RejectionSamplingStrategy()`）は、すべての要件が満たされているかどうかをチェックします。
いずれかの要件が失敗した場合、サンプリング戦略はLLMから新しいメールをサンプリングします。
このプロセスは、再試行の`loop_budget`が消費されるか、すべての要件が満たされるまで繰り返されます。

再試行があっても、サンプリングはすべての要件を満たす結果を生成しない可能性があります（`email_candidate.success==False`）。
Melleaは、LLM呼び出しが失敗することの意味について考えることを強制します。
この場合、最初のサンプルを最終結果として返すことで状況を処理します。

> [!NOTE]
> `return_sampling_results=True`パラメータを使用すると、`instruct()`関数は`ModelOutputThunk`ではなく`SamplingResult`オブジェクトを返します。これは、各サンプルのサンプリングと検証結果の完全な履歴を保持します。

### 要件の検証

要件とサンプリングを定義したので、**要件がどのように検証されるか**を見てみましょう。デフォルトの検証戦略は[LLM-as-a-judge](https://arxiv.org/abs/2306.05685)です。

要件定義をカスタマイズする方法を見てみましょう：

```python
# file: https://github.com/generative-computing/mellea/blob/main/docs/examples/tutorial/instruct_validate_repair.py#L1-L10
from mellea.stdlib.requirements import req, check, simple_validate

requirements = [
    req("The email should have a salutation"),  # == r1
    req("Use only lower-case letters", validation_fn=simple_validate(lambda x: x.lower() == x)),  # == r2
    check("Do not mention purple elephants.")  # == r3
]
```
ここで、最初の要件（r1）は、指示の出力（最後のターン）に対してLLM-as-a-judgeによって検証されます。これは、他に何も指定されていないため、デフォルトの動作です。

2番目の要件（r2）は、サンプリングステップの出力を受け取り、検証の成功/失敗を示すブール値を返す関数を単純に使用します。`validation_fn`パラメータは完全なセッションコンテキストで検証を実行する必要がありますが（第7章を参照）、Melleaは、この場合のように出力文字列を受け取りブール値を返すより単純な検証関数のラッパー（`simple_validate(fn: Callable[[str], bool])`）を提供します。

3番目の要件は`check()`です。チェックは検証にのみ使用され、生成には使用されません。
チェックは、モデル（および人間）に「Bについて考えないでください」という効果を避けることを目的としています。
これはしばしば、反対のことをして「B」について「考える」ようにプライミングします。

> [!NOTE]
> LLMaJは本質的に堅牢ではありません。可能な限り、プレーンなPythonコードを使用して要件検証を実装してください。モデルが必要な場合、検証問題専用に**キャリブレーションされた**モデルをトレーニングすることがしばしば良いアイデアです。第6章では、Melleaの`m tune`サブコマンドを使用して、要件チェック用（および他のタイプのMelleaコンポーネント用）の独自のLoRAをトレーニングする方法を説明します。


### Instruct - Validate - Repair

それでは、**instruct-validate-repair**パターンを使用した最初の生成的プログラムにすべてをまとめましょう：

```python
# file: https://github.com/generative-computing/mellea/blob/main/docs/examples/tutorial/instruct_validate_repair.py#L13-L37
import mellea
from mellea.stdlib.requirements import req, check, simple_validate
from mellea.stdlib.sampling import RejectionSamplingStrategy

def write_email(m: mellea.MelleaSession, name: str, notes: str) -> str:
    email_candidate = m.instruct(
        "Write an email to {{name}} using the notes following: {{notes}}.",
        requirements=[
            req("The email should have a salutation"),  # == r1
            req(
                "Use only lower-case letters",
                validation_fn=simple_validate(lambda x: x.lower() == x),
            ),  # == r2
            check("Do not mention purple elephants."),  # == r3
        ],
        strategy=RejectionSamplingStrategy(loop_budget=5),
        user_variables={"name": name, "notes": notes},
        return_sampling_results=True,
    )
    if email_candidate.success:
        return str(email_candidate.result)
    else:
        return email_candidate.sample_generations[0].value


m = mellea.start_session()
print(write_email(m, "Olivia",
                  "Olivia helped the lab over the last few weeks by organizing intern events, advertising the speaker series, and handling issues with snack delivery."))
```

> [!NOTE]
> `instruct()`メソッドは、`Instruction` Componentを作成してから生成する便利な関数です。`req()`も同様に`Requirement` Componentをラップします。第2章では、`m.instruct()`を呼び出したときに内部で何が起こるかを1レベル深く理解します。


### ModelOptions

ほとんどのLLM APIでは、リクエストを変更するためのオプションを指定できます：temperature、max_tokens、seedなど。Melleaは、バックエンドの初期化時およびセッションレベルの関数を`model_options`パラメータで呼び出すときに、これらのオプションを指定することをサポートしています。

Melleaは多くの異なるタイプの推論エンジン（ollama、openai互換vllm、huggingfaceなど）をサポートしています。これらの推論エンジン（`Backend`と呼びます）は、モデルオプションを指定するための異なる、時には一貫性のない辞書キーセットを提供します。モデルプロバイダー間で最も一般的なオプションについて、Melleaはエンジンに依存しないオプションを提供しており、お気に入りのIDEで[`ModelOption.<TAB>`](../mellea/backends/types.py)と入力することで使用できます。例えば、temperatureは`{ModelOption.TEMPERATURE: 0}`として指定でき、これはすべての推論エンジンで「そのまま動作」します。

バックエンドがサポートする任意のキーと値のペアを`model_options`辞書に追加でき、そのオプションは推論エンジンに渡されます。*Mellea固有の`ModelOption.<KEY>`がそのオプションに対して定義されている場合でも*。これは、既存のコードベースからモデルオプションパラメータをそのままコピーできることを意味します：

```python
# file: https://github.com/generative-computing/mellea/blob/main/docs/examples/tutorial/model_options_example.py#L1-L16
import mellea
from mellea.backends import ModelOption
from mellea.backends.ollama import OllamaModelBackend
from mellea.backends import model_ids

m = mellea.MelleaSession(backend=OllamaModelBackend(
    model_id=model_ids.IBM_GRANITE_3_2_8B,
    model_options={ModelOption.SEED: 42}
))

answer = m.instruct(
    "What is 2x2?",
    model_options={
        "temperature": 0.5,
        "num_predict": 5,
    },
)

print(str(answer))
```

特定のバックエンドのモデルオプションをいつでも更新できます。ただし、Melleaは指定されたオプションを変更するためのいくつかの追加アプローチを提供します。

1. **`m.*`呼び出し時にオプションを指定する**。ここで指定されたオプションは、その呼び出しのみで以前に指定されたモデルオプションを更新します。既存のキーを指定した場合（`ModelOption.OPTION`バージョンまたは特定のAPIのネイティブ名のいずれかで）、値は新しいキーに関連付けられた値になります。同じキーを異なる方法で指定した場合（つまり、`ModelOption.TEMPERATURE`と`temperature`）、`ModelOption.OPTION`キーが優先されます。

```python
# バックエンド初期化時に渡されるオプション
backend_model_options = {
    "seed": "1",
    ModelOption.MAX_NEW_TOKENS: 1,
    "temperature": 1,
}

# m.*時に渡されるオプション
instruct_model_options = {
    "seed": "2",
    ModelOption.SEED: "3",
    "num_predict": 2,
}

# モデルプロバイダーAPIに渡されるオプション
final_options = {
    "temperature": 1,
    "seed": 3,
    "num_predict": 2
}
```

2. **モデル状態のプッシュとポップ**。セッションはモデル状態をプッシュおよびポップする機能を提供します。これは、一連の呼び出しに対して`model_options`を一時的に変更し、ポップでそれらの変更を元に戻すことができることを意味します。

#### システムメッセージ
Melleaでは、`ModelOption.SYSTEM_PROMPT`がプロンプトのシステムメッセージを追加/変更するための推奨方法です。バックエンド/セッションレベルで設定すると、提供されたメッセージが今後のすべての呼び出しのシステムプロンプトとして使用されます（他のモデルオプションと同様）。同様に、任意のセッションレベル関数（`m.instruct`など）のシステムプロンプトパラメータを指定して、その呼び出しのみで置き換えることができます。

Melleaがこの方法でシステムメッセージを適用することを推奨するのは、一部のモデルプロバイダーAPIが`system`ロールを持つメッセージを適切にシリアライズせず、別のパラメータとして期待するためです。

### まとめ

シンプルな「Hello, World」の例から、最初の生成的プログラミングデザインパターンである**Instruct - Validate - Repair (IVR)**まで作業を進めてきました。

LLMがうまく機能する場合、ソフトウェア開発者はLLMをほとんどすべての入力を処理し、十分に望ましい出力を生成できる一種のオラクルとして経験します。LLMがまったく機能しない場合、ソフトウェア開発者はLLMをゴミを生成する単純なマルコフ連鎖として経験します。どちらの場合も、LLMは単に分布からサンプリングしているだけです。

生成的プログラミングの核心は、ほとんどのアプリケーションがこれら2つの極端の間のどこかにあることです - LLMはほとんど機能し、魅力的なMVPをデモするのに十分です。しかし、失敗モードは十分に一般的で深刻であるため、完全な自動化は開発者の手の届かないところにあります。

従来のソフトウェアは、何が間違う可能性があるかを慎重に説明し、正確なエラー処理ロジックを提供することで失敗モードに対処します。しかし、LLMを扱う場合、このアプローチはシーシュポスの呪いに苦しみます。常にもう1つの失敗モード、もう1つの特殊なケース、もう1つの新機能リクエストがあります。次の章では、構成可能で優雅に成長する生成的プログラムを構築する方法を探ります。

## 第3章：標準ライブラリの概要

さらに進む前に、Melleaのアーキテクチャを概観する必要があります。

Melleaのコア抽象化は`Component`と呼ばれます。`Component`は、LLMとのインタラクションの単位を表す構造化されたオブジェクトです。Mellea `stdlib`には有用なコンポーネントのセットが含まれていますが、独自のものを定義することもできます。すでにいくつかのコンポーネントを見てきました - `Instruction`と`Requirement`はどちらも`Component`です。

Componentsは複合データ構造です。つまり、`Component`は多くの他の部分で構成できます。これらの各部分は、`CBlock`または別の`Component`のいずれかです。`CBlock`、または「コンテンツブロック」は、テキストまたはデータの原子単位です。CBlocksは生のテキスト（または時には解析された表現）を保持し、Component DAGのリーフとして使用できます。

Componentsは、解析関数とともに期待される出力タイプを指定することもできます。デフォルトでは、このタイプは文字列です。しかし、Componentの期待されるタイプを定義することで、標準ライブラリの出力に対する型ヒントを取得できます。

Backendsは、実際にLLMを実行するエンジンです。BackendsはComponentsを消費し、Componentをフォーマットし、フォーマットされた入力をLLMに渡し、モデル出力を返します。これらはその後、CBlocksまたはComponentsに解析されます。

LLMとのインタラクション中に、いくつかのComponentsとCBlocksが作成される可能性があります。これらのインタラクションのトレースを処理するロジックは、`Context`オブジェクトによって提供されます。ContextsがComponentsとCBlocksのトレースを適切に処理するためには、いくつかの簿記が必要です。`mellea.start_session()`によって作成される`MelleaSession`クラスは、この簿記を行い、ContextsとBackendsの単純なラッパーです。

`m.instruct()`を呼び出すと、`MelleaSession.instruct`メソッドは`Instruction`と呼ばれるコンポーネントを作成します。InstructionsはMellea標準ライブラリの一部です。

これまで、説明と要件を持つInstructionsを見てきましたが、Instructionはin-context learning examplesとgrounding_context（RAG用）も持つことができます：

```python
class Instruction(Component):
    """The Instruction in an instruct/validate/repair loop."""

    def __init__(
        self,
        description: str | CBlock | None = None,
        requirements: list[Requirement | str] | None = None,
        icl_examples: list[str | CBlock] | None = None,
        grounding_context: dict[str, str | CBlock | Component] | None = None,
        user_variables: dict[str, str] | None = None,
        prefix: str | CBlock | None = None,
        output_prefix: str | CBlock | None = None,
    ):
```

次のチートシートは、Components/CBlocks、Backends、Contexts、Sessionsの関係を簡潔に視覚化しています。

TODO INSERT HENDRIK'S CHEAT SHEET

Mの標準ライブラリには、4つの基本的なタイプのComponentsが含まれています：

1. [Instructions](#第2章melleaで生成的プログラミングを始める)、すでに見てきました。
2. [Requirements](#第2章melleaで生成的プログラミングを始める)、すでに見てきて、チュートリアルの残りの部分で引き続き頻繁に使用します。
3. Generative Slots（第4章）、LLM呼び出しを関数として扱います。
4. MObjects（第5章）、ツールとそのツールが最も合理的に操作するデータを隣接させることで、ツール使用のためのコンテキストエンジニアリングを支援します。

これは可能なコンポーネントタイプの網羅的なリストではありません。新しいコンポーネントは、ユーザーライブラリまたはstdlib貢献として作成できます。意味がある場合、Componentタイプで特に動作するように設計されたファインチューニングされたモデルで新しいコンポーネントをバックアップすることもできます。

しかし、これらの高度なモダリティに入る前に、Melleaに付属する標準ライブラリのComponentsの概要を終えましょう。

---

**注記：** このファイルには第1章から第3章までの翻訳が含まれています。残りの章（第4章から第12章および付録）の翻訳が必要な場合は、お知らせください。


## 第4章：Generative Slots

古典的なプログラミングでは、純粋な（ステートレスな）関数はシンプルで強力な抽象化です。純粋な関数は入力を受け取り、出力を計算し、副作用がありません。生成的プログラムも抽象化境界として関数を使用できますが、生成的プログラムでは、関数の意味はインタープリタやコンパイラではなくLLMによって与えられます。これが**GenerativeSlot**の背後にあるアイデアです。

`GenerativeSlot`は、実装がLLMによって提供される関数です。Melleaでは、`@generative`デコレータを使用してこれらを定義します。関数シグネチャがインターフェースを指定し、docstring（または型アノテーション）がLLMに出力を生成する際のガイドを提供します。

#### 例：感情分類器

シンプルな例から始めましょう：文字列の感情を「positive」または「negative」として分類する関数です。

```python
# file: https://github.com/generative-computing/mellea/blob/main/docs/examples/tutorial/sentiment_classifier.py#L1-L13
from typing import Literal
from mellea import generative, start_session

@generative
def classify_sentiment(text: str) -> Literal["positive", "negative"]:
  """Classify the sentiment of the input text as 'positive' or 'negative'."""
  ...

m = start_session()
sentiment = classify_sentiment(m, text="I love this!")
print("Output sentiment is:", sentiment)
```

ここで、`classify_sentiment`はGenerativeSlotです：通常の関数のように見えますが、その実装はLLMによって処理されます。型アノテーション（`Literal["positive", "negative"]`）が出力を制約し、プロンプトは関数シグネチャとdocstringから自動的に構築されます。

generative slotsのより多くの例は、`docs/examples`ディレクトリに提供されています。

> [!NOTE]
> Generative slotsは、ブラックボックスの構造化出力ジェネレーターではなく、コード生成呼び出しとして実装することもできます。これは、動的解析（つまり、ランタイム情報）なしでは正しいコード生成が困難な場合に最も有用です。これらの場合、問題はランタイム状態の一部で拡張されたFiTMコード生成リクエストでプロンプトすることで解決できます。この高度な機能は信頼できないコード実行につながる可能性があるため、サンドボックス化および/または実行前の人間による検証と組み合わせて慎重に使用する必要があります。

#### Generative Slotsを使用したモジュール境界を越えた合成性の提供

Instruct-validate-repairは、特定のモジュール内で合成性を提供します。上記の例が示すように、generative slotsも同じことができます。しかし、generative slotsは単なるローカルな妥当性についてではありません。その真の力は、独立して設計されたシステム間の安全な相互運用性から来ています。

次の2つの独立して開発されたライブラリを考えてみましょう：さまざまなタイプのドキュメントを要約する関数のセットを含む**Summarizer**ライブラリと、特定の状況での意思決定を支援する**Decision Aids**ライブラリです。

```python
# file: https://github.com/generative-computing/mellea/blob/main/docs/examples/tutorial/compositionality_with_generative_slots.py#L1-L18
from mellea import generative

# The Summarizer Library
@generative
def summarize_meeting(transcript: str) -> str:
  """Summarize the meeting transcript into a concise paragraph of main points."""

@generative
def summarize_contract(contract_text: str) -> str:
  """Produce a natural language summary of contract obligations and risks."""

@generative
def summarize_short_story(story: str) -> str:
  """Summarize a short story, with one paragraph on plot and one paragraph on broad themes."""
```

```python
# file: https://github.com/generative-computing/mellea/blob/main/docs/examples/tutorial/compositionality_with_generative_slots.py#L20-L33
from mellea import generative

# The Decision Aides Library
@generative
def propose_business_decision(summary: str) -> str:
  """Given a structured summary with clear recommendations, propose a business decision."""

@generative
def generate_risk_mitigation(summary: str) -> str:
  """If the summary contains risk elements, propose mitigation strategies."""

@generative
def generate_novel_recommendations(summary: str) -> str:
  """Provide a list of novel recommendations that are similar in plot or theme to the short story summary."""
```

これら2つのライブラリは必ずしも常に構成されるわけではないことに注意してください - 会議メモには、リスク分析が意味をなすセマンティックコンテンツが含まれている場合と含まれていない場合があります。

これらのライブラリを構成するのを助けるために、関数の構成をゲートする契約のセットを導入し、それらの契約を使用してライブラリコンポーネントの無意味な構成をショートサーキットします：

```python
# file: https://github.com/generative-computing/mellea/blob/main/docs/examples/tutorial/compositionality_with_generative_slots.py#L36-L52
from mellea import generative
from typing import Literal

# Compose the libraries.
@generative
def has_structured_conclusion(summary: str) -> Literal["yes", "no"]:
  """Determine whether the summary contains a clearly marked conclusion or recommendation."""

@generative
def contains_actionable_risks(summary: str) -> Literal["yes", "no"]:
  """Check whether the summary contains references to business risks or exposure."""

@generative
def has_theme_and_plot(summary: str) -> Literal["yes", "no"]:
  """Check whether the summary contains both a plot and thematic elements."""
```

会議の議事録の例とその使用方法については、元のチュートリアルを参照してください。

これらのHoare型の契約がなければ、構成を保証する唯一の方法は、ライブラリを結合することです。契約を使用すると、結合ロジックを事前条件と事後条件のチェックに移動することで、安全な動的構成を犠牲にすることなくライブラリを**分離**できます。これは、LLMネイティブなソフトウェアエンジニアリングパターンのもう1つです：**保護された非決定性**。

## 第5章：MObjects

オブジェクト指向プログラミング（OOP）は、コードを整理するための強力なパラダイムです：関連するデータとそのデータを操作するメソッドをクラスにグループ化します。LLMの世界では、特に構造化データとLLM駆動の「ツール」や操作を組み合わせたい場合に、同様の組織原則が現れます。ここでMelleaの**MObject**抽象化が登場します。

**MObjectパターン：** データをその関連する操作（ツール）と一緒に保存する必要があります。これにより、LLMは統一された構造化された方法でデータとメソッドの両方と対話できます。また、LLMにアクセスさせたい特定のフィールドとメソッドのみを公開するプロセスが簡素化されます。

`MObject`パターンは、既存の古典的なコードベースを生成的プログラムに進化させる方法も提供します。Melleaの`@mify`デコレータを使用すると、**任意の**クラスを`MObject`に変換できます。必要に応じて、含めるフィールドとメソッドを指定し、オブジェクトがLLMにどのように表現されるかのテンプレートを提供できます。

### 例：MObjectとしてのテーブル

売上データのテーブルがあり、LLMにそれについて質問させたいとします：

```python
# file: https://github.com/generative-computing/mellea/blob/main/docs/examples/tutorial/table_mobject.py#L1-L31
import mellea
from mellea.stdlib.components.mify import mify, MifiedProtocol
import pandas
from io import StringIO


@mify(fields_include={"table"}, template="{{ table }}")
class MyCompanyDatabase:
  table: str = """| Store      | Sales   |
                    | ---------- | ------- |
                    | Northeast  | $250    |
                    | Southeast  | $80     |
                    | Midwest    | $420    |"""

  def transpose(self):
    pandas.read_csv(
      StringIO(self.table),
      sep='|',
      skipinitialspace=True,
      header=0,
      index_col=False
    )


m = mellea.start_session()
db = MyCompanyDatabase()
assert isinstance(db, MifiedProtocol)
answer = m.query(db, "What were sales for the Northeast branch this month?")
print(str(answer))
```

この例では、`@mify`デコレータがMyCompanyDatabaseをMObjectに変換します。`fields_include`で指定されているように、*table*フィールドのみがLLMプロンプトに組み込まれます。`template`は、オブジェクトがモデルにどのように提示されるかを記述します。`.query()`メソッドにより、データについて質問を投げかけることができ、LLMはテーブルをコンテキスト情報として利用できます。


**MObjectsをいつ使用するか？**
MObjectsは、構造化データとLLM駆動の操作をリンクするための洗練されたモジュラーアプローチを提供します。LLMがアクセスできるものを正確に制御でき、カスタムツールやメソッドの公開が可能です。このデザインパターンは、ツール呼び出し、ドキュメントクエリ、およびデータをLLMがアクセス可能な動作で「ラップ」する必要があるあらゆるシナリオで特に有用です。

リッチテキストドキュメントの操作に関する次のケーススタディでは、ツール登録やカスタム操作を含む、MObjectsのより高度な使用方法を見ていきます。

### ケーススタディ：ドキュメントの操作

Melleaはドキュメントの操作を簡単にします。そのために、[docling](https://github.com/docling-project/docling)ドキュメントの周りに`mified`ラッパーを提供しています。

arxiv論文からRichDocumentを作成しましょう：

```python
# file: https://github.com/generative-computing/mellea/blob/main/docs/examples/tutorial/document_mobject.py#L1-L3
from mellea.stdlib.components.docs.richdocument import RichDocument
rd = RichDocument.from_document_file("https://arxiv.org/pdf/1906.04043")
```
これはPDFファイルをロードし、Doclingパーサーを使用して中間表現に解析します。

リッチドキュメントから、例えば最初のテーブルなど、いくつかのドキュメントコンテンツを抽出できます：
```python
# file: https://github.com/generative-computing/mellea/blob/main/docs/examples/tutorial/document_mobject.py#L5-L8
from mellea.stdlib.components.docs.richdocument import Table
table1: Table = rd.get_tables()[0]
print(table1.to_markdown())
```

`Table`オブジェクトはMellea対応で、LLMですぐに使用できます。動作させてみましょう：
```python
# file: https://github.com/generative-computing/mellea/blob/main/docs/examples/tutorial/document_mobject.py#L10-L24
from mellea.backends import ModelOption
from mellea import start_session

m = start_session()
for seed in [x*12 for x in range(5)]:
    table2 = m.transform(table1,
                         "Add a column 'Model' that extracts which model was used or 'None' if none.",
                         model_options={ModelOption.SEED: seed})
    if isinstance(table2, Table):
        print(table2.to_markdown())
        break
    else:
        print(f"==== TRYING AGAIN after non-useful output.====")
```
この例では、`table1`は`Feature`列からモデル文字列を抽出する追加の列`Model`を持つように変換されるべきです（存在しない場合は`None`）。いくつかのseed値を反復処理して、テーブルの解析可能な表現を返すバージョンを見つけようとします。

モデルはタスクを見事に達成し、解析可能な構文で戻ってきました。これで、（例：`m.query(table2, "Are there any GPT models referenced?")`）を呼び出したり、変換を続けたり（例：`m.transform(table2, "Transpose the table.")`）できます。


### MObjectメソッドはツール

オブジェクトが`mified`されると、docstringを持つすべてのメソッドがLLM呼び出しのツールとして登録されます。2つのパラメータ（`funcs_include`と`funcs_exclude`）を使用して、これらの関数のサブセットのみを公開したい場合を制御できます：
```python
from mellea.stdlib.components.mify import mify

@mify(funcs_include={"from_markdown"})
class MyDocumentLoader:
    def __init__(self) -> None:
        self.content = ""

    @classmethod
    def from_markdown(cls, text: str) -> "MyDocumentLoader":
        doc = MyDocumentLoader()
        # Your parsing functions here.
        doc.content = text
        return doc

    def do_hoops(self) -> str:
        return "hoop hoop"
```
上記では、`mified`クラス`MyDocumentLoader`は`from_markdown()`メソッドのみをLLMへのツールとして公開します。


メソッドがLLM呼び出しでどのように処理されるかの例を次に示します。同じ結果につながるはずの次の2つの呼び出しを想像してください：
```python
table1_t = m.transform(table1, "Transpose the table.") # the LLM function
table1_t2 = table1.transpose() # the table method
```
`Table`のすべてのネイティブ関数は、transform関数のツールとして自動的に登録されます。つまり、ここで`.transform()`関数はLLMを呼び出し、LLMは結果を達成するために独自の`.transpose()`関数を使用することを提案して戻ってきます - また、transform関数を使用する代わりに関数呼び出しを直接使用できるという親切な警告も表示されます。


## 第6章：RequirementsとComponentsのチューニング

生成的プログラミングの主要な原則の1つは、モデルがアライメントされたのと同じ方法でモデルにプロンプトを与えるべきだということです。しかし、既製のモデルでは不十分な場合があります。私たちが遭遇したいくつかのシナリオを以下に示します：

 * 既存のモデルのトレーニングデータでは十分にカバーされていない、自明ではないセマンティクスを持つカスタムComponentを導入している
 * 既製のモデルが重要なビジネス制約を認識できない
 * 分類、意図検出、または他の要件のようなタスクを改善するために使用したい独自のラベル付きデータセットがある

3番目のケースは非常に一般的です。このチュートリアルでは、そのケースに焦点を当てたケーススタディを探ります。分類データを使用してLoRAアダプターをファインチューニングし、要件チェッカーを強化する方法を説明します。次に、このファインチューニングされたアダプターをMelleaプログラムに組み込む方法を説明します。

### 問題提起

[第4章](#第4章generative-slots)で遭遇したStembolt MFG Corporationは現在、運用効率と回復力を向上させるためのAIエージェントを開発しています。このパイプラインの重要なコンポーネントはAutoTriageモジュールです。AutoTriageは、自由形式の欠陥レポートをmini-carburetor、piston、connecting rod、flywheel、piston rings、no_failureなどのカテゴリに自動的にマッピングする責任があります。

生成された出力が特定のダウンストリームシステム要件を満たすことを保証するために、各欠陥サマリーに特定された故障モードが含まれている必要があります。残念ながら、LLMはこのタスクを箱から出してすぐには不十分に実行します。ステムボルトはニッチなデバイスであり、検出レポートはオープンインターネット上で一般的に議論されていません。幸いなことに、長年にわたり、Stembolt MFGはメモを部品故障にマッピングする大規模なデータセットを収集しており、これがaLoRAを介してトレーニングされた分類器が登場する場所です。

Stembolt MFGの慎重に作成された[ステムボルト故障モードのデータセット](https://github.com/generative-computing/mellea/blob/main/docs/examples/aLora/stembolt_failure_dataset.jsonl)の小さなサブセットを以下に示します：

```json
{"item": "Observed black soot on intake. Seal seems compromised under thermal load.", "label": "piston rings"}
{"item": "Rotor misalignment caused torsion on connecting rod. High vibration at 3100 RPM.", "label": "connecting rod"}
{"item": "Combustion misfire traced to a cracked mini-carburetor flange.", "label": "mini-carburetor"}
{"item": "stembolt makes a whistling sound and does not complete the sealing process", "label": "no_failure"}
```

最後の項目が「no_failure」とラベル付けされていることに注意してください。これは、その問題の根本原因がユーザーエラーであるためです。ステムボルトは使用が難しく、専門的なトレーニングが必要です。報告された故障の約20％は実際にはオペレーターエラーです。オペレーターエラーをプロセスのできるだけ早い段階で、十分な精度で分類することは、Stembolt部門のカスタマーサービスおよび修理部門にとって重要なKPIです。

Stembolt MFG Corporationが、より大きなMelleaアプリケーションでAutoTriageステップを実装するためにチューニングされたLoRAをどのように使用できるかを見てみましょう。

### aLoRAアダプターのトレーニング

Melleaは、[LoRA](https://arxiv.org/abs/2106.09685)または[aLoRA](https://github.com/huggingface/peft/blob/main/docs/source/developer_guides/lora.md#activated-lora-alora)アダプターをトレーニングするためのコマンドラインインターフェースを提供します。古典的なLoRAは、コンテキスト全体を再処理する必要があり、内部ループ内で発生する迅速なチェック（要件チェックなど）では高コストになる可能性があります。aLoRAメソッドを使用すると、ベースLLMを新しいタスクに適応させ、最小限の計算オーバーヘッドでアダプターを実行できます。アダプターはトレーニングが速く、切り替えも高速です。

この小さなデータセットで`m alora train`コマンドを使用して軽量アダプターをトレーニングします：

> [!NOTE]
> このスクリプトを実行するにはGPUへのアクセスが必要です。CPUでも実行できますが、時間がかかる場合があります。
> Macユーザーの場合、accelerateライブラリで`fp16`サポートがないため、このスクリプトをそのまま実行できない可能性があります。

```bash
m alora train /to/stembolts_data.jsonl \
  --promptfile ./prompt_config.json \
  --basemodel ibm-granite/granite-3.2-8b-instruct \
  --outfile ./checkpoints/alora_adapter \
  --adapter alora \
  --epochs 6 \
  --learning-rate 6e-6 \
  --batch-size 2 \
  --max-length 1024 \
  --grad-accum 4
```

デフォルトのプロンプト形式は`<|start_of_role|>check_requirement<|end_of_role|>`です。このプロンプトは、新しくトレーニングされたaLoRAをアクティブ化する直前にコンテキストに追加する必要があります。必要に応じて、`--promptfile`引数を使用してこのプロンプトをカスタマイズできます。

#### パラメータ

アダプターをトレーニングする際、以下のようにハイパーパラメータを簡単に調整できます：

| フラグ              | タイプ    | デフォルト   | 説明                                      |
|-------------------|---------|-----------|--------------------------------------------------|
| `--basemodel`     | `str`   | *必須*| Hugging Face モデルIDまたはローカルパス              |
| `--outfile`       | `str`   | *必須*| アダプター重みを保存するディレクトリ            |
| `--adapter`       | `str`   | `"alora"` | `alora`または標準`lora`から選択        |
| `--epochs`        | `int`   | `6`       | トレーニングエポック数                        |
| `--learning-rate` | `float` | `6e-6`    | 学習率                                    |
| `--batch-size`    | `int`   | `2`       | デバイスごとのバッチサイズ                            |
| `--max-length`    | `int`   | `1024`    | 最大トークン化入力長                       |
| `--grad-accum`    | `int`   | `4`       | 勾配累積ステップ                      |
| `--promptfile`    | `str`   | None      | プロンプト形式をロードするディレクトリ              |


### Hugging Faceへのアップロード（オプション）

トレーニングされたアダプターを共有または再利用するには、`m alora upload`コマンドを使用してトレーニングされたアダプターを公開します：

```bash
m alora upload ./checkpoints/alora_adapter \
  --name stembolts/failuremode-alora
```

これにより：
- Hugging Faceモデルリポジトリを作成します（存在しない場合）
- `outfile`ディレクトリの内容をアップロードします
- `huggingface-cli login`を介した有効な`HF_TOKEN`が必要です

権限エラーが発生した場合は、Huggingfaceにログインしていることを確認してください：

```bash
huggingface-cli login  # オプション：アップロードにのみ必要
```

> [!NOTE]
> **プライバシーに関する警告：** トレーニングされたモデルをHugging Face Hubにアップロードする前に、可視性を慎重に確認してください。モデルを一般に公開する場合は、トレーニングデータに独自の、機密の、または機密情報が含まれているかどうかを検討してください。言語モデルは意図せず詳細を記憶する可能性があり、この問題は小規模またはドメイン固有のデータセットで操作する場合に悪化します。


### チューニングされたモデルのMelleaへの統合

タスク用のaLoRA分類器をトレーニングした後、その分類器を使用してMelleaプログラムで要件をチェックしたいと思います。まず、aLoRA分類器を使用するためにバックエンドをセットアップする必要があります：

```python
backend = ...

# 前提：`m`バックエンドは、aloraをトレーニングした同じベースモデルを持つHuggingfaceまたはalora互換vLLMバックエンドである必要があります。
# ollamaはまだLoRAまたはaLoRAアダプターをサポートしていません。

backend.add_alora(
    HFConstraintAlora(
        name="stembolts_failuremode_alora",
        path_or_model_id="stembolts/failuremode-alora", # チェックポイントパスでも可
        generation_prompt="<|start_of_role|>check_requirement<|end_of_role|>",
        backend=m.backend,
    )
)
```

上記の引数で、`path_or_model_id`は前のステップのモデルチェックポイント、つまり`m alora train`プロセスを指します。

> [!NOTE]
> `backend.add_alora`呼び出しに渡される`generation_prompt`は、トレーニングに使用されたプロンプトと正確に一致する必要があります。

これで、Mセッションを作成し、要件を定義し、指示を実行する準備が整いました：

```python
m = MelleaSession(backend, ctx=ChatContext())
failure_check = req("The failure mode should not be none.")
res = m.instruct("Write triage summaries based on technician note.", requirements=[failure_check])
```

要件がよくトレーニングされたaloraモデルでうまく機能するようにするには、要件検証関数も定義する必要があります：

```python
def validate_reqs(reqs: list[Requirement]):
    """Validate the requirements against the last output in the session."""
    print("==== Validation =====")
    print(
        "using aLora"
        if backend.default_to_constraint_checking_alora
        else "using NO alora"
    )

    # helper to collect validation prompts (because validation calls never get added to session contexts).
    logs: list[GenerateLog] = []  # type: ignore

    # Run the validation. No output needed, because the last output in "m" will be used. Timing added.
    start_time = time.time()
    val_res = m.validate(reqs, generate_logs=logs)
    end_time = time.time()
    delta_t = end_time - start_time

    print(f"Validation took {delta_t} seconds.")
    print("Validation Results:")

    # Print list of requirements and validation results
    for i, r in enumerate(reqs):
        print(f"- [{val_res[i]}]: {r.description}")

    # Print prompts using the logs list
    print("Prompts:")
    for log in logs:
        if isinstance(log, GenerateLog):
            print(f" - {{prompt: {log.prompt}\n   raw result: {log.result.value} }}")  # type: ignore

    return end_time - start_time, val_res
```

次に、この検証関数を使用して、生成された欠陥レポートを次のようにチェックできます：

```python
validate_reqs([failure_check])
```

制約aloraがモデルに追加されている場合、デフォルトで使用されます。aloraなしで実行を強制することもできます：

```python
backend.default_to_constraint_checking_alora = False
```

この章では、分類データセットを使用して独自データでLoRAアダプターをチューニングする方法を見てきました。次に、結果のモデルをMellea生成的プログラムに組み込む方法を見ました。これは非常に大きな氷山の一角です。


## 第7章：コンテキスト管理について

Melleaは2つの補完的なメカニズムを使用してコンテキストを管理します：

1. `Component`自体。これは一般的に単一ターンのリクエストに必要なすべてのコンテキストを含みます。MObjectsはフィールドとメソッドを使用してコンテキストを管理し、InstructionsにはRAGスタイルのリクエスト用のgrounding_contextがあります。

2. `Context`。これは現在のセッション中にLLMに対して行われたすべての以前のリクエストの（時には部分的な）履歴を保存および表現します。

ComponentsをLLMリクエストのコンテキストを定義するために使用する方法についてはすでに多く見てきたので、この章では`Context`メカニズムに焦点を当てます。

`start_session()`メソッドを使用すると、実際にはデフォルトの推論エンジン、デフォルトのモデル選択、およびデフォルトのコンテキストマネージャーを持つ`Mellea`をインスタンス化しています。次のコードは`m.start_session()`と同等です：

```python
from mellea import MelleaSession

m = mellea.MelleaSession(
    backend=OllamaBackend(model_id=IBM_GRANITE_3_3_8B)
    context=SimpleContext()
)
```

`SimpleContext` - これまで使用してきた唯一のコンテキスト - は、各モデル呼び出しでチャットメッセージ履歴をリセットするコンテキストマネージャーです。つまり、モデルのコンテキストは現在のComponentによって完全に決定されます。Melleaは、チャット履歴のように動作する`ChatContext`も提供します。ChatContextを使用してチャットモデルと対話できます：

```python
# file: https://github.com/generative-computing/mellea/blob/main/docs/examples/tutorial/context_example.py#L1-L5
from mellea import start_session

m = mellea.start_session(ctx=ChatContext())
m.chat("Make up a math problem.")
m.chat("Solve your math problem.")
```

`Context`オブジェクトは、現在のモデルコンテキストを内省するためのいくつかの有用なヘルパーを提供します。例えば、常に最後のモデル出力を取得できます：

```python
# file: https://github.com/generative-computing/mellea/blob/main/docs/examples/tutorial/context_example.py#L7
print(m.ctx.last_output())
```

または、最後のターン全体（ユーザークエリ + アシスタント応答）を取得できます：

```python
# file: https://github.com/generative-computing/mellea/blob/main/docs/examples/tutorial/context_example.py#L9
print(m.ctx.last_turn())
```

`session.clone()`を使用して、特定の時点でのコンテキストを持つセッションのコピーを作成することもできます。これにより、コンテキスト内の同じオブジェクトで複数の生成リクエストを行うことができます：
```python
m = start_session(ctx=ChatContext())
m.instruct("Multiply 2x2.")

m1 = m.clone()
m2 = m.clone()

# Need to run this code in an async event loop.
co1 = m1.ainstruct("Multiply that by 3")
co2 = m2.ainstruct("Multiply that by 5")

print(await co1)  # 12
print(await co2)  # 20
```
上記の例では、両方のリクエストのコンテキストに`Multiply 2x2`とそれに対するLLMの応答（おそらく`4`）があります。セッションをクローンすることで、新しいリクエストは両方ともそのコンテキスト上で独立して動作し、4 x 3と4 x 5の正しい答えを得ます。

## 第8章：エージェントの実装

> **定義：** *エージェント*とは、LLMがプログラムの制御フローを決定する生成的プログラムです。

これまで見てきた生成的プログラムでは、開発者がLLM呼び出しのシーケンスを調整します。対照的に、エージェント的な生成的プログラムは、制御フローをモデル自体に委任します。この章では、Melleaでエージェントを開発するいくつかの異なる方法を見ていきます：

1. **古典的なエージェント：** ReACTパターンを使用してMelleaでエージェントループを実装する方法。
2. **保護された非決定性：** generative slotsのアイデアに戻り、この抽象化がより堅牢なエージェントの構築にどのように役立つかを見ます。

### ケーススタディ：MelleaでReACTを実装する

ReACTパターンを使用した完全なエージェントの例を構築しましょう。疑似コードから始めて、Mellea ReACTプログラムを段階的に構築します。

ReACTの核心的なアイデアは、推論（「Thought」）と行動（「Action」）を交互に行うことです：

```
# Pseudocode
while not done:
    get the model's next thought
    take an action based upon the thought
    choose arguments for the selected action
    observe the tool output
    check if a final answer can be obtained
return the final answer
```

このエージェントがMelleaでどのように実装されているかを見てみましょう：

```python
# file: https://github.com/generative-computing/mellea/blob/main/docs/examples/agents/react.py#L99
def react(
        m: mellea.MelleaSession,
        goal: str,
        react_toolbox: ReactToolbox,
        budget: int = 5,
):
    assert m.ctx.is_chat_context, "ReACT requires a chat context."
    test_ctx_lin = m.ctx.render_for_generation()
    assert (
            test_ctx_lin is not None and len(test_ctx_lin) == 0
    ), "ReACT expects a fresh context."

    # Construct the system prompt for ReACT.
    _sys_prompt = react_system_template.render(
        {"today": datetime.date.today(), "tools": react_toolbox.tools}
    )

    # Add the system prompt and the goal to the chat history.
    m.ctx.insert(mellea.stdlib.chat.Message(role="system", content=_sys_prompt))
    m.ctx.insert(mellea.stdlib.chat.Message(role="user", content=f"{goal}"))

    done = False
    turn_num = 0
    while not done:
        turn_num += 1
        print(f"## ReACT TURN NUMBER {turn_num}")

        print(f"### Thought")
        thought = m.chat(
            "What should you do next? Respond with a description of the next piece of information you need or the next action you need to take."
        )
        print(thought.content)

        print("### Action")
        act = m.chat(
            "Choose your next action. Respond with a nothing other than a tool name.",
            format=react_toolbox.tool_name_schema(),
        )
        selected_tool: ReactTool = react_toolbox.get_tool_from_schema(
            act.content)
        print(selected_tool.get_name())

        print(f"### Arguments for action")
        act_args = m.chat(
            "Choose arguments for the tool. Respond using JSON and include only the tool arguments in your response.",
            format=selected_tool.args_schema(),
        )
        print(
            f"```json\n{json.dumps(json.loads(act_args.content), indent=2)}\n```")

        # TODO: handle exceptions.
        print("### Observation")
        tool_output = react_toolbox.call_tool(selected_tool, act_args.content)
        m.ctx.insert(
            mellea.stdlib.chat.Message(role="tool", content=tool_output)
        )
        print(tool_output)

        is_done = IsDoneModel.model_validate_json(
            m.chat(
                f"Do you know the answer to the user's original query ({goal})? If so, respond with Yes. If you need to take more actions, then respond No.",
                format=IsDoneModel,
            ).content
        ).is_done
        if is_done:
            print("Done. Will summarize and return output now.")
            done = True
            return m.chat(
                f"Please provide your final answer to the original query ({goal})."
            ).content
        elif turn_num == budget:
            return None
```

### 保護された非決定性

第4章を思い出してください。そこでは、`GenerativeSlot`コンポーネントのライブラリが合成性契約を導入することでどのように構成できるかを見ました。ここでは、構成可能な可能性のある生成的関数を連鎖させるタスクを自動化するための「エージェント的な」メカニズムを構築します。保護された非決定性エージェント（「guarded nondeterminism」は少し長いので、今後は[Kripke](https://en.wikipedia.org/wiki/Saul_Kripke)エージェントと呼びます）を始めましょう。

最初のステップは、generative slotsに事前条件と事後条件を追加する新しい`Component`を追加することです：

```python
# file: https://github.com/generative-computing/kripke_agents/blob/main/kripke/base.py#L10-L38
class ConstrainedGenerativeSlot(Component):
    template = GEN_SLOT_TEMPLATE # the same template as is used for generative slots.

    def __init__(self, generative_slot: GenerativeSlot, preconds: list[Requirement | str], postconds: list[Requirement | str]):
        self._genslot = generative_slot
        self._preconds = [reqify(precond) for precond in preconds]
        self._postconds = [reqify(postcond) for postcond in postconds]

    def format_for_llm(self):
        return self._genslot.format_for_llm()

    def action_name(self):
        return self._genslot._function._function_dict["name"]
```

便利なデコレータも追加します：

```python
# file: https://github.com/generative-computing/kripke_agents/blob/main/kripke/base.py#L41-L44
def constrained(preconds: list[Requirement | str], postconds: list[Requirement | str]):
    def _decorator(genslot: GenerativeSlot):
        return ConstrainedGenerativeSlot(genslot, preconds, postconds)
    return _decorator
```

これで、次のように制約付きgenerative slotsを記述できます：

```python
# file: https://github.com/generative-computing/kripke_agents/blob/main/main.py#L23-L27
@constrained(preconds=["contains a summary of the story's theme"], postconds=["each element of the list is the title and author of a significant novel"])
@generative
def suggest_novels_based_on_theme(summary: str) -> list[str]:
    """Based upon a summary of a short story, suggests novels with similar themes."""
    ...
```

`Requirement`コンポーネントを全体で使用していることに注意してください。これにより、事前条件/事後条件を定義およびチェックするためにMelleaの要件検証セマンティクスのすべての力を利用できます。

これで、kripkeエージェントのスタンプを提供する準備が整いました：

```python
# file: https://github.com/generative-computing/kripke_agents/blob/main/kripke/base.py#L54-L99
def filter_actions(m: mellea.MelleaSession, actions: list[ConstrainedGenerativeSlot], *, output: ModelOutputThunk | None = None):
  ...

## 第9章：他のフレームワークとの相互運用性

Melleaプログラムは、結局のところ、単なるPythonプログラムです。MelleaプログラムはModel Context ProtocolまたはA2Aプロトコルを介して共有できます。Melleaプログラムは、これらのプロトコルを実装するツールやエージェントを消費することもできます。

### Melleaを実行するシンプルなmcpサーバー

前述のように、melleaは最終的にはpythonプログラムです。プログラムの周りにシンプルな`mcp`サーバーをラップして、サーバーをそのまま使用できます。以下は、[Pydantic AIの組み込みmcpサーバー](https://ai.pydantic.dev/mcp/server/)を使用した例です。

```python
# file: https://github.com/generative-computing/mellea/blob/main/docs/examples/agents/mcp_example.py#L15-L40
# Create an MCP server
mcp = FastMCP("Demo")


@mcp.tool()
def write_a_poem(word_limit: int) -> str:
    """Write a poem with a word limit."""
    m = MelleaSession(OllamaModelBackend(model_ids.QWEN3_8B))
    wl_req = Requirement(
        f"Use only {word_limit} words.",
        validation_fn=simple_validate(lambda x: len(x.split(" ")) < word_limit),
    )

    res = m.instruct(
        "Write a poem",
        requirements=[wl_req],
        strategy=RejectionSamplingStrategy(loop_budget=4),
    )
    assert isinstance(res, ModelOutputThunk)
    return str(res.value)

if __name__ == '__main__':
    mcp.run()
```

### Melleaプログラムをopenai互換サーバーとして実行（実験的）

OpenAI互換の**chat**エンドポイントを提供するための実験的な`m serve`ユーティリティも提供しています。これにより、「モデル」として振る舞う`m`プログラムを書くことができます。この機能の詳細については、次を実行してください：

```shell
m serve --help
```

#### 例：`m serve`アプリケーション

`m serve`を使用してプログラムをデプロイする場合、プログラムが特定の構造に従うことが重要です。プログラムには、次のシグネチャを持つ`serve`という関数が必要です：

```python
# file: https://github.com/generative-computing/mellea/blob/main/docs/examples/agents/m_serve_example.py#L25-L29
def serve(
    input: list[ChatMessage],
    model_options: None | dict = None,
    **kwargs
)
```

`m serve`コマンドは、この関数を受け取り、openai互換のサーバーを実行します。詳細については、`m serve`互換プログラムの書き方について[このファイル](./examples/tutorial/m_serve_example.py)を参照してください。例を実行するには：

```shell
m serve docs/examples/tutorial/m_serve_example.py
```

## 第10章：Mのためのプロンプトエンジニアリング

ほとんどのバックエンドはテキストで動作します。これらのバックエンド/モデルの場合、MelleaはPythonオブジェクトをテキストに変換する方法について独自の立場を持っています：`TemplateFormatter`です。

ほとんどの場合、標準ライブラリに新しいコンポーネントを追加する場合、または新しいモデル用に既存のコンポーネントをカスタマイズする場合に、テンプレートを作成する必要があります。

### テンプレート
Melleaの`TemplateFormatter`は、jinja2テンプレートを使用して、生成のためにモデルに渡すときにオブジェクトをフォーマットします。

これらのテンプレートは、クラス/オブジェクトに直接保存することも、より一般的には、各オブジェクトが特定のファイルを持つディレクトリに保存することもできます。テンプレートの例については、`mellea/templates/prompts/default`を参照してください。

フォーマッターがどのテンプレートを使用するかを選択する方法の説明については、以下の[カスタマイズセクション](#カスタマイズ)を参照してください。

### テンプレート表現
テンプレートとともに、各クラス/オブジェクトは、テンプレートをレンダリングするときに提供される引数を定義する必要があります。これは、コンポーネントの`format_for_llm()`関数で行われます。これは、文字列または`TemplateRepresentation`のいずれかを返します。

`string`：最も単純なアプローチは、このメソッドがオブジェクトの文字列表現を返すことです。これにより、テンプレート化が完全に回避されます。

`TemplateRepresentation`：`TemplateRepresentation`オブジェクトを返すこともできます。
この表現には以下が含まれます：
    - コンポーネントへの参照
    - テンプレートレンダラーに渡される引数の辞書
    - コンポーネントに関連するツール/関数のリスト

また、次のフィールドのいずれかも含まれます：
- template：提供された引数でレンダリングできるjinja2テンプレートの文字列表現
- template_order：検索するテンプレートファイルの名前を記述する文字列のリスト（「.jinja2」サフィックスなし）。`*`はクラス名を示します。

### カスタマイズ
新しいテンプレートを書いたり、コンポーネントのTemplateRepresentationを変更したりすることで、テキスト表現をカスタマイズできます。モデルに基づいてカスタマイズすることもできます。

#### テンプレートの選択
コンポーネントのTemplateRepresentationに`template_order`フィールドが含まれていると仮定すると、デフォルトのTemplateFormatterは、`template_order`内の各テンプレートについて、次の場所を順番に調べて関連するテンプレートを取得します：
1. テンプレートが最近検索された場合、フォーマッターのキャッシュされたテンプレート
2. フォーマッターの指定されたテンプレートパス
3. フォーマットされるオブジェクトが属するパッケージ（'mellea'またはサードパーティパッケージ）


## 第11章：ツール呼び出し

Melleaは、それをサポートするプロバイダー/モデルのツール呼び出しをサポートしています。ほとんどのセッションレベル関数は、tool_callsブール値の設定をサポートしています。これをtrueに設定すると、ツールを呼び出すことができますが、モデルがそれらを呼び出す保証はありません。

ツールをモデルが呼び出せるようにするには、いくつかの方法があります：
1. Components：コンポーネントは、ツールを含むTemplateRepresentationオブジェクトを持つことができます。
2. Context：コンテキストに応じて、そのコンテキスト内のコンポーネントは、現在のアクションである場合とまったく同じ方法で追加ツールのソースとして使用できます。
3. `ModelOptions.TOOLS`：モデルオプションにはツールパラメータを含めることができます。これらのツールを渡す推奨方法は、関数オブジェクトのリストとしてです。

現在、ツールは関数の名前によって識別されます。競合がある場合、その名前を持つ最新のツールが優先されます。これは、モデルが利用できるツールが上記と同じ優先順位を持つことを意味します：
1. 現在のコンポーネントからのツールは常に含まれます
2. コンテキストからのツールは、名前の競合がない場合に含まれます。特定のコンテキストは、どのツールを表示するかを決定できますが、ほとんどの場合、コンテキスト内の最新のコンポーネントからのツールは、古いリクエストからのツールよりも優先されます。
3. `ModelOptions.TOOLS`からのツールは、上記の関数のいずれとも競合しない場合にのみ追加されます。

コンポーネントのテンプレート表現にツールを追加する例については、[richdocument.py](../mellea/stdlib/docs/richdocument.py)の`Table`オブジェクトを参照してください。

モデルオプションを介してツールを追加する例を次に示します。これは、ほぼ常に利用可能であるべきWeb検索のようなツールを追加する場合に役立ちます：
```python
import mellea
from mellea.backends import ModelOption

def web_search(query: str) -> str:
    ...

m = mellea.start_session()
output = m.instruct(
    "Who is the 1st President of the United States?",
    model_options={
        ModelOption.TOOLS: [web_search],
    },
    tool_calls = True,
)

assert "web_search" in output.tool_calls

result = output.tool_calls["web_search"].call_func()
```

## 第12章：非同期性

Melleaは、非同期関数と同期関数内の非同期イベントループという、いくつかの方法で非同期動作をサポートしています。

### 非同期関数

`MelleaSession`には、Pythonの通常の非同期関数と同じように動作する非同期関数があります。これらの非同期セッション関数は、同期版をミラーリングしています：
```python
m = start_session()
result = await m.ainstruct("Write your instruction here!")
```

ただし、複数の非同期関数を同時に実行する場合は、コンテキストに注意する必要があります。デフォルトでは、`MelleaSession`は履歴のない`SimpleContext`を使用します。これは、複数の非同期リクエストを一度に実行する場合に問題なく動作します：
```python
m = start_session()
coroutines = []

for i in range(5):
    coroutines.append(m.ainstruct(f"Write a math problem using {i}"))

results = await asyncio.gather(*coroutines)
```

`ChatContext`を使用しようとする場合、コンテキストが適切に変更されるように、各リクエスト間でawaitする必要があります：
```python
m = start_session(ctx=ChatContext())

result = await m.ainstruct("Write a short fairy tale.")
print(result)

main_character = await m.ainstruct("Who is the main character of the previous fairy tail?")
print(main_character)
```

そうしないと、リクエストは期待するメッセージを持たない古いコンテキストを使用します。例えば：
```python
m = start_session(ctx=ChatContext())

co1 = m.ainstruct("Write a very long math problem.")  # Start first request.
co2 = m.ainstruct("Solve the math problem.")  # Start second request with an empty context.

results = await asyncio.gather(co1, co2)
for result in results:
    print(result)  # Neither request had anything in its context.

print(m.ctx)  # Only shows the operations from the second request.
```

さらに、これらのコンテキストの問題を回避するために`session.clone()`を使用する方法の例については、[第7章：コンテキスト管理](#第7章コンテキスト管理について)を参照してください。

### 同期関数内の非同期性

Melleaは内部的に非同期性を利用します。`m.instruct`を呼び出すと、結果を生成するためにLLMへの非同期リクエストを実行する同期コードを使用しています。単一のリクエストの場合、これは実行速度に違いをもたらしません。

`SamplingStrategy`を使用する場合や検証中に、Melleaは複数の結果を生成し、それらの結果を複数の要件に対して同時に検証することで、プログラムの実行時間を高速化できます。`m.instruct`または非同期の`m.ainstruct`のどちらを使用しても、Melleaはリクエストをできるだけ早くディスパッチし、非同期に結果を待つことで、リクエストを高速化しようとします。

## 付録：Melleaへの貢献

### 貢献者ガイド：はじめに

Melleaに貢献する場合は、precommitフックを使用することが重要です。これらのフックを使用する、またはテストスイートを実行するには、`[all]`オプション依存関係とdevグループをインストールする必要があります。

```
git clone git@github.com:generative-computing/mellea.git && 
cd mellea && 
uv venv .venv && 
source .venv/bin/activate &&
uv pip install -e ".[all]" --group dev
pre-commit install
```

その後、`pytest`を実行してすべてのテストを実行するか、`CICD=1 pytest`を実行してCI/CDテストのみを実行できます。特定のテストカテゴリ（例：バックエンド別、リソース要件別）の実行の詳細については、[test/MARKERS_GUIDE.md](../test/MARKERS_GUIDE.md)を参照してください。

ヒント：`git commit`に`-n`フラグを渡すことでフックをバイパスできます。これは、後でスカッシュする予定の中間コミットに役立つことがあります。

### 貢献者ガイド：RequirementsとVerifiers

新しいRequirements（つまり、検証器）を貢献することは、Melleaへの貢献を始める簡単な方法です。Requirementsは、一般的またはドメイン固有のいずれでもかまいませんが、一貫性のあるテスト可能なプロパティをカプセル化する必要があります。このチュートリアル全体でRequirementsの多くの例を見てきました。

汎用的で他の人にも役立つ可能性が高いRequirementを書いた場合は、*汎用*コンポーネントをMelleaの標準ライブラリに貢献することを検討してください：

1. `mellea/stdlib/reqlib/`で要件が属するファイルを見つけます。適合するファイルがない場合は、新しいファイルを作成します。
2. 要件を実装します。理想的には、検証器は堅牢であるべきで、これは通常、デフォルトのLLMaJ動作を使用しないことを意味します。要件がコードでチェックできる場合は、検証関数を書く必要があります。これがどのように機能するかの例については、[Markdown requirements](../mellea/stdlib/reqlib/md.py)を参照してください。コードで実装できない要件については、[よくキャリブレーションされたaLoRAをチューニング（および評価）](#第6章requirementsとcomponentsのチューニング)することもできます。
3. PRを開きます。RequirementがLLMaJを使用する場合は、LLMaJ検証が十分であることを示す堅牢な評価スイートをPRに含めてください。

重要な注意事項：要件が文法的制約の観点から簡単に指定できる場合は、要件を使用する代わりに、制約付き生成を使用することを検討してください（セッションまたは生成呼び出しに`format=`を渡すことで - いくつかの例については[エージェント実装の章](#第8章エージェントの実装)を参照）。

### 貢献者ガイド：Components

Componentsは、Melleaの構成要素です。Componentのポイントは、Backendに自分自身を表現する方法、つまり`format_for_llm`関数を持つことです。新しいコンポーネントを作成する場合、`format_for_llm`が`TemplateRepresentation`を返すようにしたいと思うでしょう。これは、テンプレート引数、ツール、およびテンプレート自体を含む構造化された表現です。

Componentsは、LLMと対話するためにテキストに頻繁にフォーマットおよびマーシャリングしているデータ/オブジェクトがある場合に最適に作成されます。

新しいコンポーネントを作成するには、コードで定義し、（ほとんどの場合）そのためのテンプレートを作成する必要があります。Componentsはランタイムチェック可能なプロトコルでもあるため、ベースクラスから継承する必要はありません。既存のクラスに必要なメソッドを追加するだけでもかまいません。

新しいComponentを配布する場合、Componentをソフトウェアライブラリと同じように考えてください。Componentsは自己完結型で、よく文書化されており、再利用可能で、できれば他のComponentsと構成可能です。

Componentを配布するには、いくつかのオプションがあります。Componentをユーザースペースのライブラリとして配布することも、ComponentをMellea stdlibに組み込むようにリクエストすることもできます。ほとんどのComponentsは、サードパーティライブラリとして最適に配置されます。サードパーティの生成的プログラミングコンポーネントは、他のサードパーティライブラリ（github、pypi）を配布するのと同じように配布できます。

有用で広く使用されているパターンを実装するComponentsの場合、Mellea stdlibへの包含が意味をなす可能性があります。これらは生成的プログラミングの初期の日々です。一部の貢献がMellea標準ライブラリで誇りの場所を持つことを期待しています。貢献者には、stdlibへの包含について早期かつ頻繁に尋ねることをお勧めします。

### 貢献者ガイド：特殊化されたMify

オブジェクトをmifyすることは、`Mellea`と互換性を持たせるもう1つの方法です。Componentsと同様に、ランタイムチェック可能なプロトコルである`MifiedProtocol`があります。`@mify`または`mify(object)`は、任意のオブジェクトに必要なメソッドを追加します。

これはプロトコルであるため、クラス/オブジェクトをラップするか、そのクラス/オブジェクトに必要な機能を任意の方法で追加する独自の`mify`関数を作成できます。

たとえば、ほとんどのオブジェクトが同じパターンと構造に従うORMライブラリがある場合があります。そのライブラリを`Mellea`と統合するには、その構造について知っている特定の`mify`関数を書くというアプローチがあります。次のようになります：
```python
T = TypeVar("T")
def mify_orm(obj: T):
  setattr(obj, "format_for_llm", obj.sql)
  ...
```
この方法で、すべてが`sql`関数を持つと仮定して、このライブラリのすべてのコンポーネントを即座に`mify`する共通の方法を定義できます。

特殊化されたmify関数をstdlibに追加するには、デコレータとしても、オブジェクト/クラスで直接呼び出せる関数としても機能する必要があります。また、汎用的だが有用なパターン、または広く使用されているライブラリのパターンである必要があります。

### 貢献者ガイド：Sessions

あまり一般的ではないニーズですが、Melleaでは新しいタイプのセッションを作成できます。コンテキストをきめ細かく制御する必要がある場合は、`MelleaSession`メソッドを完全にオーバーライドすることをお勧めします。

基礎となるコンテキストを変更せずに、行われる呼び出しにゲートを設定したり、呼び出しを変更したりするには、メソッドをオーバーライドしながら`MelleaSession`スーパーメソッドを呼び出すことをお勧めします。[`chat-checker`の例](./examples/sessions/creating_a_new_type_of_session.py)を参照してください。
