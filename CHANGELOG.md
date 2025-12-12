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
