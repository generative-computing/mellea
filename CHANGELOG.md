## [v0.3.1](https://github.com/generative-computing/mellea/releases/tag/v0.3.1) - 2026-02-11

### Feature

* Migrate from Granite 3 to Granite 4 hybrid models ([#357](https://github.com/generative-computing/mellea/issues/357)) ([`8f9e18c`](https://github.com/generative-computing/mellea/commit/8f9e18caf5e7cce38a0e3634eb07c8d98ba520db))
* Add MelleaTool.from_smolagents() for smolagents integration ([#430](https://github.com/generative-computing/mellea/issues/430)) ([`0471006`](https://github.com/generative-computing/mellea/commit/0471006ae174744a194f1e43129802ed749c15bb))
* Add tool calling argument validation ([#364](https://github.com/generative-computing/mellea/issues/364)) ([`840a02d`](https://github.com/generative-computing/mellea/commit/840a02d877d7a147f5db14f2c57492d9074fbb15))
* Instrument telemetry ([#355](https://github.com/generative-computing/mellea/issues/355)) ([`b2e5a52`](https://github.com/generative-computing/mellea/commit/b2e5a5288717535708aa53fb72a4fd958c29056e))
* Add query clarification RAG intrinsic support ([#391](https://github.com/generative-computing/mellea/issues/391)) ([`d38698a`](https://github.com/generative-computing/mellea/commit/d38698a4c71ed49c71e8d9efe0420a379185f3dd))
* Add mellea react agent ([#402](https://github.com/generative-computing/mellea/issues/402)) ([`7884b8d`](https://github.com/generative-computing/mellea/commit/7884b8dbb3985ec03a3c8bd10aa82dbf0e932ffc))
* Optimize example test discovery and execution speed ([#372](https://github.com/generative-computing/mellea/issues/372)) ([`e9aefaf`](https://github.com/generative-computing/mellea/commit/e9aefaf7b066179d8062076c41db650569e92d43))
* New MelleaTool class and adoption across mellea ([#380](https://github.com/generative-computing/mellea/issues/380)) ([`ffb8b6c`](https://github.com/generative-computing/mellea/commit/ffb8b6c797b20ec387a2c2257e88900b4e498f0f))
* Add code coverage tracking with pytest-cov ([#353](https://github.com/generative-computing/mellea/issues/353)) ([`b45a4b6`](https://github.com/generative-computing/mellea/commit/b45a4b6a214de9e4f632ef34f51fecade6abf3b6))
* Add pytest markers for test categorization (#322) ([#326](https://github.com/generative-computing/mellea/issues/326)) ([`0d8d020`](https://github.com/generative-computing/mellea/commit/0d8d02052192ea0153792ecd2e1ee9fa764b18b5))

### Fix

* Lint/format issues ([#536](https://github.com/generative-computing/mellea/issues/536)) ([`781bb6b`](https://github.com/generative-computing/mellea/commit/781bb6b7d8028611c5ea66c702f4c5f2f52eccbc))
* Tools in examples ([#535](https://github.com/generative-computing/mellea/issues/535)) ([`a49bdf8`](https://github.com/generative-computing/mellea/commit/a49bdf8a7f450a22a9c337b64f007b5e07e205ea))
* Quick fix to get the role / content from specifically parsed messages ([#533](https://github.com/generative-computing/mellea/issues/533)) ([`2f54cc8`](https://github.com/generative-computing/mellea/commit/2f54cc8daf5fbdca0d47420ff54f944416428364))
* Migrate from IBM alora to PEFT 0.18.1 native aLoRA ([#422](https://github.com/generative-computing/mellea/issues/422)) ([`c6a3e64`](https://github.com/generative-computing/mellea/commit/c6a3e643b092b5d4f09f7003d6a9c347d92d8052))
* Flag more tests that require ollama ([#420](https://github.com/generative-computing/mellea/issues/420)) ([`b06851f`](https://github.com/generative-computing/mellea/commit/b06851fd4d1134bae0ad253536a994269327c2f6))
* Guarantee proper ordering of decompose subtask dependencies ([#407](https://github.com/generative-computing/mellea/issues/407)) ([`f0b1346`](https://github.com/generative-computing/mellea/commit/f0b1346d3e7aafdd2ca83979bfb723df081255d5))
* Astream output ([#358](https://github.com/generative-computing/mellea/issues/358)) ([`9cafe05`](https://github.com/generative-computing/mellea/commit/9cafe053a6fc1ec2fa90fd5c35be12527f4cda45))
* Update ci for merge-queue ([#417](https://github.com/generative-computing/mellea/issues/417)) ([`5cf8eee`](https://github.com/generative-computing/mellea/commit/5cf8eee83cc46e04defa070a6d0f31d43dbb001e))
* Some examples needed update ([#408](https://github.com/generative-computing/mellea/issues/408)) ([`3d5ab56`](https://github.com/generative-computing/mellea/commit/3d5ab56eeb6760e8608289cf5d9de989bb727f24))
* Formatting model_ids for better readability ([#386](https://github.com/generative-computing/mellea/issues/386)) ([`318a962`](https://github.com/generative-computing/mellea/commit/318a9623f6da046c9547cd41816933f113b853e3))
* Update agents.md to strongly encourage using uv ([#388](https://github.com/generative-computing/mellea/issues/388)) ([`8b2e2cf`](https://github.com/generative-computing/mellea/commit/8b2e2cfa6b48b39016a11e5d3839abf8dc93a2a4))
* Restrict transformers version to 4.x ([#379](https://github.com/generative-computing/mellea/issues/379)) ([`67f8bc0`](https://github.com/generative-computing/mellea/commit/67f8bc075ebdf061da29ee5db6caba81e786aa7a))
* Friendly error messages for optional backend dependencies ([#343](https://github.com/generative-computing/mellea/issues/343)) ([`4f7091f`](https://github.com/generative-computing/mellea/commit/4f7091f1a2b22143705ce3b87e893722efba40f1))
* Add missing await keywords in async tests ([#346](https://github.com/generative-computing/mellea/issues/346)) ([`a7442a6`](https://github.com/generative-computing/mellea/commit/a7442a6fd799a676beb4c290dfc9c93f3ef9d47e))
* Use __repr__ for helpful debug display in Message/ToolMessage ([#339](https://github.com/generative-computing/mellea/issues/339)) ([`f15fadb`](https://github.com/generative-computing/mellea/commit/f15fadb3cfa5d47fab2bb7283eb5a648c397188e))
* Add skip to timeout test for python < 3.11 ([#333](https://github.com/generative-computing/mellea/issues/333)) ([`2cc3352`](https://github.com/generative-computing/mellea/commit/2cc3352473bdde9b44e1e533125ee71e0dbc0ab2))
* Don't overwrite user-configured logging levels ([#298](https://github.com/generative-computing/mellea/issues/298)) ([`119ea86`](https://github.com/generative-computing/mellea/commit/119ea86d9da3d74bd966c9757315cf3a0c724087))

### Documentation

* Bedrock example. ([#410](https://github.com/generative-computing/mellea/issues/410)) ([`3204b3a`](https://github.com/generative-computing/mellea/commit/3204b3a82877eabe83451f8038df80334058c623))
* Add decompose to tutorial with example ([#366](https://github.com/generative-computing/mellea/issues/366)) ([`ef0a964`](https://github.com/generative-computing/mellea/commit/ef0a96499346dd3cddc20ca461a8490c4164cfca))
* Create contributing doc ([#369](https://github.com/generative-computing/mellea/issues/369)) ([`1cacbf9`](https://github.com/generative-computing/mellea/commit/1cacbf9c2fbb4c5e8d035683e99c8f04671824b4))
* Add security policy ([#363](https://github.com/generative-computing/mellea/issues/363)) ([`afbda1d`](https://github.com/generative-computing/mellea/commit/afbda1d5dc90dd9c9d0cfadbc7785c0656389141))
* Add code of conduct ([#365](https://github.com/generative-computing/mellea/issues/365)) ([`94b21d9`](https://github.com/generative-computing/mellea/commit/94b21d97ae7215605f7fb0170b3abdb930b34b66))
* Add discord badge to readme ([#362](https://github.com/generative-computing/mellea/issues/362)) ([`168ccca`](https://github.com/generative-computing/mellea/commit/168ccca6d2d7f13ee5c8ad7ab28bac7c58222cb6))

### Performance

* Use module-scoped fixture for RAG tests ([#337](https://github.com/generative-computing/mellea/issues/337)) ([`d72ecfd`](https://github.com/generative-computing/mellea/commit/d72ecfd280051a679b26683e7b170458033238e2))

## [v0.3.0](https://github.com/generative-computing/mellea/releases/tag/v0.3.0) - 2026-01-21

### Feature

* SOFAI Sampling Strategy ([#311](https://github.com/generative-computing/mellea/issues/311)) ([`cbf3913`](https://github.com/generative-computing/mellea/commit/cbf3913845f0186fd89c7ca5d1bd0e29f3ac33b7))
* Reorg of codebase ([#310](https://github.com/generative-computing/mellea/issues/310)) ([`cbc456b`](https://github.com/generative-computing/mellea/commit/cbc456b242f9aa74b74f2a672dd58126b4672fd3))
* Add typed components; add typing to model output thunks and sampling results ([#300](https://github.com/generative-computing/mellea/issues/300)) ([`2eb689d`](https://github.com/generative-computing/mellea/commit/2eb689d019a8eb7f89c748a986156a4813b08147))

### Fix

* Tool calling code sample in tutorial ([#313](https://github.com/generative-computing/mellea/issues/313)) ([`a42a487`](https://github.com/generative-computing/mellea/commit/a42a4873522445713380d4a2a690c22a1ad13e89))
* Adds granite-common[transformers] to Mellea's huggingface depedency group. ([#330](https://github.com/generative-computing/mellea/issues/330)) ([`87a8166`](https://github.com/generative-computing/mellea/commit/87a81664dc9b65da3b643651a7d5a7c5d9656b84))
* Rename file from test_* ([#332](https://github.com/generative-computing/mellea/issues/332)) ([`6512b32`](https://github.com/generative-computing/mellea/commit/6512b329a506a263592fd033ca90ef0a6d0a5557))
* Readd init file for mellea/stdlib ([#328](https://github.com/generative-computing/mellea/issues/328)) ([`cb156a7`](https://github.com/generative-computing/mellea/commit/cb156a73231e3fedc7dc6e742fbf85e7ea0c2838))
* ImageBlocks are CBlocks ([#323](https://github.com/generative-computing/mellea/issues/323)) ([`8a4c910`](https://github.com/generative-computing/mellea/commit/8a4c910e48b72845c75fc14a46dcd34ddee2aeda))
* Additional tests optimization when running on github actions. ([#293](https://github.com/generative-computing/mellea/issues/293)) ([`c5398e4`](https://github.com/generative-computing/mellea/commit/c5398e4af4120ae7722dde7787b63269ce10daf3))
* Import times by not exporting RichDocument at module level ([#321](https://github.com/generative-computing/mellea/issues/321)) ([`565b27f`](https://github.com/generative-computing/mellea/commit/565b27f381c01db3f1d11f480e10275b2889ea57))
* Add logging for start_session details ([#299](https://github.com/generative-computing/mellea/issues/299)) ([`6e68f57`](https://github.com/generative-computing/mellea/commit/6e68f5749c59802821edb71f01d7f9249fe1db6f))
* Typos in READMEs and documentation ([#303](https://github.com/generative-computing/mellea/issues/303)) ([`9f6a086`](https://github.com/generative-computing/mellea/commit/9f6a0860857e36fe873885f7078412b982d3f5e8))
* Add explicit exports to __init__.py ([#317](https://github.com/generative-computing/mellea/issues/317)) ([`6e7b09b`](https://github.com/generative-computing/mellea/commit/6e7b09b6b7bceac7529d69012f00cb252fd815a6))
* Mify protocol issues ([#304](https://github.com/generative-computing/mellea/issues/304)) ([`7013b04`](https://github.com/generative-computing/mellea/commit/7013b04513858b4a740b69187a42bce486747fcc))
* Linting error ([#302](https://github.com/generative-computing/mellea/issues/302)) ([`c6e3b08`](https://github.com/generative-computing/mellea/commit/c6e3b08c7f6f9fcd52b22975f7f2f03773a0b070))
* Add double quotes around brackets used in pip install ([#301](https://github.com/generative-computing/mellea/issues/301)) ([`2d017f1`](https://github.com/generative-computing/mellea/commit/2d017f186e52839246c69dc528656cafab79543e))

### Documentation

* Add AGENTS.md to guide AI coding assistants ([#320](https://github.com/generative-computing/mellea/issues/320)) ([`a89256a`](https://github.com/generative-computing/mellea/commit/a89256a6a826b00a403b58abcb889ae730881421))
* Improve contributor instructions in README. ([#314](https://github.com/generative-computing/mellea/issues/314)) ([`2be67c8`](https://github.com/generative-computing/mellea/commit/2be67c8a85c8ad4ee48359b2ce0ace315de9594d))

## [v0.2.4](https://github.com/generative-computing/mellea/releases/tag/v0.2.4) - 2026-01-08

### Fix

* Fix gc in instructions and add exception to generate walk ([#295](https://github.com/generative-computing/mellea/issues/295)) ([`5fc7df0`](https://github.com/generative-computing/mellea/commit/5fc7df0a0680ced342575fc4cc8787385334c7a4))
* Marks span tests as qualitative & removes chat error message. ([#294](https://github.com/generative-computing/mellea/issues/294)) ([`5ce6360`](https://github.com/generative-computing/mellea/commit/5ce63604da385550b0781e7429599b5f64a8b2b8))

## [v0.2.3](https://github.com/generative-computing/mellea/releases/tag/v0.2.3) - 2026-01-07

### Feature

* Allow forcing a release through test failures ([#292](https://github.com/generative-computing/mellea/issues/292)) ([`14b55a3`](https://github.com/generative-computing/mellea/commit/14b55a38550581e5b0e0ea05cf5cc8b45a2cd5e5))
* Lazy Spans and KV Blocks ([#249](https://github.com/generative-computing/mellea/issues/249)) ([`b9e4a33`](https://github.com/generative-computing/mellea/commit/b9e4a33d4a7c79bfe3655aaf10a957947e78d04f))
* Switch to new RAG intrinsics repo ([#289](https://github.com/generative-computing/mellea/issues/289)) ([`94c35ad`](https://github.com/generative-computing/mellea/commit/94c35adff9d44376a9357e06b20dafd09314b29b))

### Fix

* OpenAI `base_url` default and reasoning effort model option. ([#271](https://github.com/generative-computing/mellea/issues/271)) ([`9733df8`](https://github.com/generative-computing/mellea/commit/9733df8fc2e3a81a0f5ba7c35e1cf468d9eb622f))
* Unpin granite_commons version from 0.3.5 ([#287](https://github.com/generative-computing/mellea/issues/287)) ([`0b402bd`](https://github.com/generative-computing/mellea/commit/0b402bdc8f73d7543ff56bdbe1a5c304182fcbb5))

## [v0.2.2](https://github.com/generative-computing/mellea/releases/tag/v0.2.2) - 2025-12-18

### Feature

* Add langchain / message interop example ([#257](https://github.com/generative-computing/mellea/issues/257)) ([`9b1f299`](https://github.com/generative-computing/mellea/commit/9b1f29961aa15bcc2cad2181d2c598e668b4f383))
* Add better error messages for incorrect genslot args ([#248](https://github.com/generative-computing/mellea/issues/248)) ([`9d875d6`](https://github.com/generative-computing/mellea/commit/9d875d669979f58bc574853837b8b51bacd7f0db))

### Fix

* Uv-lock package changes ([#261](https://github.com/generative-computing/mellea/issues/261)) ([`cb0623f`](https://github.com/generative-computing/mellea/commit/cb0623f12a23c1e567bcc2c4f6659adb571738cb))
* Lock granite-common version to avoid arg changes ([#260](https://github.com/generative-computing/mellea/issues/260)) ([`03716c1`](https://github.com/generative-computing/mellea/commit/03716c1c6ce37be921ba47ae07ec9191021d6e49))
* Docstrings to have code blocks ([#256](https://github.com/generative-computing/mellea/issues/256)) ([`94a7b40`](https://github.com/generative-computing/mellea/commit/94a7b40950b8c25882f8acf5106d08e353e13d21))

## [v0.2.1](https://github.com/generative-computing/mellea/releases/tag/v0.2.1) - 2025-12-10

### Feature

* Test-based Evaluation with LLM-as-a-judge ([#225](https://github.com/generative-computing/mellea/issues/225)) ([`0f1f0f8`](https://github.com/generative-computing/mellea/commit/0f1f0f8eb12e60f7940e3ad5b40ce91ded73fc88))
* Add a `code_interpreter` tool ([#232](https://github.com/generative-computing/mellea/issues/232)) ([`b03c964`](https://github.com/generative-computing/mellea/commit/b03c96439501146965cd123ce5046f2c5907acfb))

### Fix

* Add simple lock to hf generation to prevent using incorrect weights ([#237](https://github.com/generative-computing/mellea/issues/237)) ([`6b2a527`](https://github.com/generative-computing/mellea/commit/6b2a5276a426be87ce02fa89f49818535f211fa6))
* Collection of small fixes ([#238](https://github.com/generative-computing/mellea/issues/238)) ([`2120112`](https://github.com/generative-computing/mellea/commit/2120112ee807da4e980eabc0df54b3aae12d2cd2))
* Fix unused litellm import ([#246](https://github.com/generative-computing/mellea/issues/246)) ([`633bfd7`](https://github.com/generative-computing/mellea/commit/633bfd7198eac45f8c163fefcc910d9bf8a76151))
* Minor updates to answer relevance ([#245](https://github.com/generative-computing/mellea/issues/245)) ([`bde9b4d`](https://github.com/generative-computing/mellea/commit/bde9b4dd91ab92af0f7a661e321bb9d701da0589))
* Pre-commit file selection ([#243](https://github.com/generative-computing/mellea/issues/243)) ([`e70d307`](https://github.com/generative-computing/mellea/commit/e70d3075f92152f8bb1a519687575fc7f30ffe1b))

### Documentation

* Fixed copyright in LICENSE ([#210](https://github.com/generative-computing/mellea/issues/210)) ([`3087051`](https://github.com/generative-computing/mellea/commit/3087051f5fc102bb8e0319af6baf9d7a0222e6ef))

## [v0.2.0](https://github.com/generative-computing/mellea/releases/tag/v0.2.0) - 2025-11-19

### Feature

* Change backend functions to use async; add generate_from_raw ([`16b8aea`](https://github.com/generative-computing/mellea/commit/16b8aea1ab4fc18428adafb2c6106314d986c537))
* Updates for intrinsics support ([#227](https://github.com/generative-computing/mellea/issues/227)) ([`52953a5`](https://github.com/generative-computing/mellea/commit/52953a507729e8683d8b027d7c1e6d70b2356955))
* Add requirements and preconditions to gen slots ([#226](https://github.com/generative-computing/mellea/issues/226)) ([`f73d8e2`](https://github.com/generative-computing/mellea/commit/f73d8e23c57146b44e8b552f5e30315e353ff592))
* MelleaSession.register for functional interface and MelleaSession.powerup for dynamic mixin (register all methods in a class) ([#224](https://github.com/generative-computing/mellea/issues/224)) ([`662cfcc`](https://github.com/generative-computing/mellea/commit/662cfcc99c365411c7dcee0d55fcd0cba21bd4b8))
* Add secure Python code execution with llm-sandbox support ([#217](https://github.com/generative-computing/mellea/issues/217)) ([`9d12458`](https://github.com/generative-computing/mellea/commit/9d12458432db3c1172d79ffdcbfae50f2bf8b402))
* Adds think budget-forcing ([#107](https://github.com/generative-computing/mellea/issues/107)) ([`a2e29e6`](https://github.com/generative-computing/mellea/commit/a2e29e633b9f470d3992335becb8231dc57d0d69))
* Making generate_from_raw public ([#219](https://github.com/generative-computing/mellea/issues/219)) ([`7eae224`](https://github.com/generative-computing/mellea/commit/7eae2244763a4349e202e6b87502d23e111ea07e))
* Conda/Mamba-based installation script ([#138](https://github.com/generative-computing/mellea/issues/138)) ([`6aea9dc`](https://github.com/generative-computing/mellea/commit/6aea9dc85b0147a22ff5a5553a75d9179958ce6e))
* Adds a vllm backend ([#122](https://github.com/generative-computing/mellea/issues/122)) ([`21908e5`](https://github.com/generative-computing/mellea/commit/21908e5bbc6bfd3bfd6f84953cefb3f6a56fccf2))
* Add the ability to run examples with pytest ([#198](https://github.com/generative-computing/mellea/issues/198)) ([`e30afe6`](https://github.com/generative-computing/mellea/commit/e30afe6148d68b6ef1d6aa3417823c7a51ff0743))
* Ollama generate_from_raw uses existing event loop ([#204](https://github.com/generative-computing/mellea/issues/204)) ([`36a069f`](https://github.com/generative-computing/mellea/commit/36a069fb6f9912a25c5c8aa51a5fe46ce2e945d3))

### Fix

* Vllm format issues ([`abbde23`](https://github.com/generative-computing/mellea/commit/abbde236d4d5900a3717d4a6af4759743dcd21d9))
* Some minor fixes ([#223](https://github.com/generative-computing/mellea/issues/223)) ([`7fa0891`](https://github.com/generative-computing/mellea/commit/7fa08915573ee696d230dffef5532be8b7d3b7e3))
* Watsonx self._project_id not getting set ([#220](https://github.com/generative-computing/mellea/issues/220)) ([`10f6ffa`](https://github.com/generative-computing/mellea/commit/10f6ffa35ea089b2396d184b18a1efbac75b94a7))
* Decomp subtask regex ([#218](https://github.com/generative-computing/mellea/issues/218)) ([`5ac34be`](https://github.com/generative-computing/mellea/commit/5ac34be51ee1d14678888d53c6374810a7ed5871))

### Documentation

* Adding pii m serve example ([#215](https://github.com/generative-computing/mellea/issues/215)) ([`54f13f4`](https://github.com/generative-computing/mellea/commit/54f13f4c0314ff21189a4a06051dfea84b5420d1))

## [v0.1.3](https://github.com/generative-computing/mellea/releases/tag/v0.1.3) - 2025-10-22

### Feature

* Decompose cli tool enhancements & new prompt_modules ([#170](https://github.com/generative-computing/mellea/issues/170)) ([`b8fc8e1`](https://github.com/generative-computing/mellea/commit/b8fc8e1bd9478d87c6a9c5cf5c0cca751f13bd11))
* Add async functions ([#169](https://github.com/generative-computing/mellea/issues/169)) ([`689e1a9`](https://github.com/generative-computing/mellea/commit/689e1a942efab6cb1d7840f6bdbd96d579bdd684))
* Add Granite Guardian 3.3 8B with updated examples function call validation and repair with reason. ([#167](https://github.com/generative-computing/mellea/issues/167)) ([`517e9c5`](https://github.com/generative-computing/mellea/commit/517e9c5fb93cba0b5f5a69278806fc0eda897785))
* Majority voting sampling strategy ([#142](https://github.com/generative-computing/mellea/issues/142)) ([`36eaca4`](https://github.com/generative-computing/mellea/commit/36eaca482957353ba505d494f7be32c5226de651))

### Fix

* Fix vllm install script ([#185](https://github.com/generative-computing/mellea/issues/185)) ([`abcf622`](https://github.com/generative-computing/mellea/commit/abcf622347bfbb3c5d97c74a2624bf8f051f4136))
* Watsonx and litellm parameter filtering ([#187](https://github.com/generative-computing/mellea/issues/187)) ([`793844c`](https://github.com/generative-computing/mellea/commit/793844c44ed091f4c6abae1cc711e3746a960ef4))
* Pin trl to version 0.19.1 to avoid deprecation ([#202](https://github.com/generative-computing/mellea/issues/202)) ([`9948907`](https://github.com/generative-computing/mellea/commit/9948907303774494fee6286d482dd10525121ba2))
* Rename format argument in internal methods for better mypiability ([#172](https://github.com/generative-computing/mellea/issues/172)) ([`7a6f780`](https://github.com/generative-computing/mellea/commit/7a6f780bdd71db0a7e0a1e78dfc78dcc4e4e5d93))
* Async overhaul; create global event loop; add client cache ([#186](https://github.com/generative-computing/mellea/issues/186)) ([`1e236dd`](https://github.com/generative-computing/mellea/commit/1e236dd15bd426ed31f148ccdca4c63e43468fd0))
* Update readme and other places with granite model and tweaks ([#184](https://github.com/generative-computing/mellea/issues/184)) ([`519a35a`](https://github.com/generative-computing/mellea/commit/519a35a7bb8a2547e90cf04fd5e70a3f74d9fc22))

## [v0.1.2](https://github.com/generative-computing/mellea/releases/tag/v0.1.2) - 2025-10-03

### Feature

* Making Granite 4 the default model ([#178](https://github.com/generative-computing/mellea/issues/178)) ([`545c1b3`](https://github.com/generative-computing/mellea/commit/545c1b3790fa96d7d1c76878227f60a2203862b4))

### Fix

* Default sampling strats to None for query, transform, chat ([#179](https://github.com/generative-computing/mellea/issues/179)) ([`c8d4601`](https://github.com/generative-computing/mellea/commit/c8d4601bad713638a2a8e1c1062e19548f182f3c))
* Docstrings ([#177](https://github.com/generative-computing/mellea/issues/177)) ([`6126bd9`](https://github.com/generative-computing/mellea/commit/6126bd922121a080a88b69718603a15bc54f80f4))
* Always call sample when a strategy is provided ([#176](https://github.com/generative-computing/mellea/issues/176)) ([`8fece40`](https://github.com/generative-computing/mellea/commit/8fece400f1483fa593c564ad70f5b7370d3dd249))

## [v0.1.1](https://github.com/generative-computing/mellea/releases/tag/v0.1.1) - 2025-10-01

### Fix

* Bump patch version to allow publishing ([#175](https://github.com/generative-computing/mellea/issues/175)) ([`cf7a24b`](https://github.com/generative-computing/mellea/commit/cf7a24b2541c081cda8f2468bb8e7474ed2618a8))

## [v0.1.0](https://github.com/generative-computing/mellea/releases/tag/v0.1.0) - 2025-10-01

### Feature

* Add fix to watsonx and note to litellm ([#173](https://github.com/generative-computing/mellea/issues/173)) ([`307dbe1`](https://github.com/generative-computing/mellea/commit/307dbe14d430b0128e56a2ed7b735dbe93adf2a7))
* New context, new sampling,. ([#166](https://github.com/generative-computing/mellea/issues/166)) ([`4ae6d7c`](https://github.com/generative-computing/mellea/commit/4ae6d7c23e4aff63a0887dccaf7c96bc9e50121a))
* Add async and streaming support ([#137](https://github.com/generative-computing/mellea/issues/137)) ([`4ee56a9`](https://github.com/generative-computing/mellea/commit/4ee56a9f9e74302cf677377d6eab19e11ab0a715))
* Best-of-N Sampling with Process Reward Models ([#118](https://github.com/generative-computing/mellea/issues/118)) ([`b18e03d`](https://github.com/generative-computing/mellea/commit/b18e03d655f18f923202acf96a49d4acafa0701d))

## [v0.0.6](https://github.com/generative-computing/mellea/releases/tag/v0.0.6) - 2025-09-18

### Feature

* Test update pypi.yml for cd pipeline test ([#155](https://github.com/generative-computing/mellea/issues/155)) ([`91003e5`](https://github.com/generative-computing/mellea/commit/91003e572ed770da5c685cbc275facddb7700da6))

## [v0.0.5](https://github.com/generative-computing/mellea/releases/tag/v0.0.5) - 2025-09-17

### Feature

* Enable VLMs ([#126](https://github.com/generative-computing/mellea/issues/126)) ([`629cd9b`](https://github.com/generative-computing/mellea/commit/629cd9be8ab5ee4227eb662ac5f73bc0c42e668c))
* LiteLLM backend ([#60](https://github.com/generative-computing/mellea/issues/60)) ([`61d7f0e`](https://github.com/generative-computing/mellea/commit/61d7f0e2e9f5e8cc756a294b0580d27ccce2aaf6))
* New logo by Ja Young Lee ([#120](https://github.com/generative-computing/mellea/issues/120)) ([`c8837c6`](https://github.com/generative-computing/mellea/commit/c8837c695e2d6a693a441e3fc9e1fabe231b11f0))

### Fix

* Adding pillow as dependency ([#147](https://github.com/generative-computing/mellea/issues/147)) ([`160c6ef`](https://github.com/generative-computing/mellea/commit/160c6ef92fc5ca352de9daa066e6f0eda426f3d9))
* Huggingface backend does not properly pad inputs ([#145](https://github.com/generative-computing/mellea/issues/145)) ([`a079c77`](https://github.com/generative-computing/mellea/commit/a079c77d17f250faaafb0cd9bcc83972c2186683))
* Return to old logo ([#132](https://github.com/generative-computing/mellea/issues/132)) ([`f08d2ec`](https://github.com/generative-computing/mellea/commit/f08d2ec8af680ffee004ba436123a013efae7063))
* Alora version and image printing in messages ([#130](https://github.com/generative-computing/mellea/issues/130)) ([`2b3ff55`](https://github.com/generative-computing/mellea/commit/2b3ff55fcfb61ef30a26365b9497b31df7339226))
* Remove ModelOption.THINKING from automatic mapping because it's explicitly handled in line #417 (which was causing parameter conflicts) ([#124](https://github.com/generative-computing/mellea/issues/124)) ([`b5c2a39`](https://github.com/generative-computing/mellea/commit/b5c2a394e3bc62961a55310aeb5944238791dbc1))

### Documentation

* Improved documentation on model_options ([#134](https://github.com/generative-computing/mellea/issues/134)) ([`ad10f3b`](https://github.com/generative-computing/mellea/commit/ad10f3bc57a6cf68777c1f78b774414935f47a92))
* Explain that the tool must be called ([#140](https://github.com/generative-computing/mellea/issues/140)) ([`a24a8fb`](https://github.com/generative-computing/mellea/commit/a24a8fbd68b986496b563a74414f3fb8b1f02355))
* Fix typo on README ([#116](https://github.com/generative-computing/mellea/issues/116)) ([`dc610ae`](https://github.com/generative-computing/mellea/commit/dc610ae427f2b18008c537ea1737130e1f062a78))
* Fix README typos and broken links ([`4d90c81`](https://github.com/generative-computing/mellea/commit/4d90c81ea916d8f38da11182f88154219181fdd1))
