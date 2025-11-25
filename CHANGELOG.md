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
