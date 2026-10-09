# Changelog

## [0.10.0](https://github.com/superlinked/sie/compare/v0.9.0...v0.10.0) (2026-10-09)


### ⚠ BREAKING CHANGES

* **server:** a LoRA whose PEFT config sets bias to "all" or "lora_only" no longer loads. A profile that declares one still loads its model without that LoRA and logs the refusal. A request that names one gets the same retryable LORA_LOADING response as for any other LoRA that fails to load in the background, and the server log carries the refusal. Retrain or export the LoRA with bias="none".

### Features

* **config:** accept hybrid encode and score routing in a cluster ([#576](https://github.com/superlinked/sie/issues/576)) ([96de71c](https://github.com/superlinked/sie/commit/96de71cd2d2ef977c0003eae8c961567efd904a5)), closes [#415](https://github.com/superlinked/sie/issues/415)
* **config:** refuse a remote profile that names an undefined upstream ([#500](https://github.com/superlinked/sie/issues/500)) ([4977097](https://github.com/superlinked/sie/commit/497709744ce5e827aa89a8191a840c6ea14b65ce))
* **examples:** add bounded agent stage latency trials ([2326f19](https://github.com/superlinked/sie/commit/2326f19b2b9a04670b7d355e72085c4b651c0842))
* **examples:** add custom-entity-types, caller-named entity types against LLMs, Comprehend and spaCy ([#468](https://github.com/superlinked/sie/issues/468)) ([7860d1f](https://github.com/superlinked/sie/commit/7860d1f28e17c899db8f011ae8b4d5ccb76ab8c1))
* **examples:** add reproducible agent stage latency evaluation ([aa20315](https://github.com/superlinked/sie/commit/aa20315ba13ac92221d3fdb348b1183f644114d4))
* **examples:** add reproducible photo transcription study ([#630](https://github.com/superlinked/sie/issues/630)) ([a826a95](https://github.com/superlinked/sie/commit/a826a9554de22dfcd420a7fa20c2dd073342b342))
* **examples:** add structured-output-accuracy, strict structured outputs scored on the Structured Output Benchmark ([#474](https://github.com/superlinked/sie/issues/474)) ([96b7f9b](https://github.com/superlinked/sie/commit/96b7f9b63d9035853851cc901fc2e146e2ad62ba))
* **examples:** compare matched short-verdict guardrails configurations ([#479](https://github.com/superlinked/sie/issues/479)) ([3ebda81](https://github.com/superlinked/sie/commit/3ebda81134cbba29330f826f7b946b8dfca32fe1))
* **examples:** document-to-markdown-olmocr prints the score without the math files ([#459](https://github.com/superlinked/sie/issues/459)) ([37e1380](https://github.com/superlinked/sie/commit/37e13808c412f851f7517247dad0907b0d4185d8))
* **examples:** evaluate ordinary named entity tags ([#629](https://github.com/superlinked/sie/issues/629)) ([4ecec8a](https://github.com/superlinked/sie/commit/4ecec8ad13c6ebb620f88ad74c633ac061a02d88))
* **examples:** rebuild guardrails as a harmful-prompt screening study ([#456](https://github.com/superlinked/sie/issues/456)) ([01e606e](https://github.com/superlinked/sie/commit/01e606e616e09bf59bbcb7c0d6ecc165f222399f))
* **examples:** rebuild image-search on a held-out Amazon Berkeley Objects product catalogue ([#470](https://github.com/superlinked/sie/issues/470)) ([8041b15](https://github.com/superlinked/sie/commit/8041b15b47ec44d92b671fc8f6c6b711bad79312))
* **examples:** rebuild redact on the pre-registered PII coverage run ([#452](https://github.com/superlinked/sie/issues/452)) ([f99850a](https://github.com/superlinked/sie/commit/f99850a7b66905b66977c15af5d189a60e1aa783))
* **examples:** rebuild visual-document-search on the ViDoRe v3 head-to-head ([#472](https://github.com/superlinked/sie/issues/472)) ([832f561](https://github.com/superlinked/sie/commit/832f5612a760c208275bc3541f3e3573e171432c))
* **examples:** replay document-field extraction evidence ([#511](https://github.com/superlinked/sie/issues/511)) ([971f2d4](https://github.com/superlinked/sie/commit/971f2d4f55684ae1d963c19bbb43e8c3399d35aa))
* **examples:** reproduce caller-rule rerank study ([#485](https://github.com/superlinked/sie/issues/485)) ([c5df8f9](https://github.com/superlinked/sie/commit/c5df8f98d2ec3f840a1bf7ed9658d2417ab8365a))
* **examples:** reproduce GLM-OCR receipt and handwriting results ([#486](https://github.com/superlinked/sie/issues/486)) ([ea15097](https://github.com/superlinked/sie/commit/ea150976ef272ba5d3ab9b70778643c5451c9e4e))
* **examples:** reproduce RF100-VL named-object detection ([#487](https://github.com/superlinked/sie/issues/487)) ([56d932e](https://github.com/superlinked/sie/commit/56d932ea60364cdf9e3961945324f4b476d71c92))
* **examples:** reproduce the image retrieval pilot ([#602](https://github.com/superlinked/sie/issues/602)) ([ed79362](https://github.com/superlinked/sie/commit/ed7936255757919b14df3f06a70899c9f6ba1526))
* **gateway:** bridge a lane its transport reports cold ([#592](https://github.com/superlinked/sie/issues/592)) ([b5d6253](https://github.com/superlinked/sie/commit/b5d6253e7fa2625fdf2a3ba8b96aca0b6a8bd920))
* **gateway:** bridge buffered cluster generation refusals ([#547](https://github.com/superlinked/sie/issues/547)) ([6ec8dec](https://github.com/superlinked/sie/commit/6ec8dec33f364dbd05d4662424c77cc4b85ea721))
* **gateway:** bridge cluster extraction and audio refusals ([#549](https://github.com/superlinked/sie/issues/549)) ([b009e9f](https://github.com/superlinked/sie/commit/b009e9f40c62a20a6f3a36ec0e5c5a4db95d1467))
* **gateway:** bridge encode and score under a numerical admission ([#587](https://github.com/superlinked/sie/issues/587)) ([f6e0820](https://github.com/superlinked/sie/commit/f6e0820715f3f3328fd4db9acfddaa25d6c5cfe4)), closes [#415](https://github.com/superlinked/sie/issues/415)
* **gateway:** bridge governed generation to the remote route a policy names ([#607](https://github.com/superlinked/sie/issues/607)) ([e875a65](https://github.com/superlinked/sie/commit/e875a65eda3a02d228498b78467445ed9a34b191))
* **gateway:** carry generation GPU time ([11612e4](https://github.com/superlinked/sie/commit/11612e4e9c6435fc69bc1f55a0c4a49cc5679dca))
* **gateway:** carry sealed generation GPU time ([2a4a3a1](https://github.com/superlinked/sie/commit/2a4a3a1bb47e7ea58860bf6cfeb0384a73077b09))
* **gateway:** carry the fallback reason on every bridged work item ([#567](https://github.com/superlinked/sie/issues/567)) ([8823a89](https://github.com/superlinked/sie/commit/8823a895228fd2611e5957e225ef6321d2823ad2))
* **gateway:** carry the requested model on queued work ([#504](https://github.com/superlinked/sie/issues/504)) ([c98bfbb](https://github.com/superlinked/sie/commit/c98bfbb038dcff59411bcb925d1bf59f70c0ef44))
* **gateway:** coordinate threshold demand with bounded leases ([#554](https://github.com/superlinked/sie/issues/554)) ([6e1424a](https://github.com/superlinked/sie/commit/6e1424a205514be969b385a88a038af07b86d3c7))
* **gateway:** disclose which side served on gateway responses ([#502](https://github.com/superlinked/sie/issues/502)) ([04e6159](https://github.com/superlinked/sie/commit/04e6159e97c058cce1d8915b84acd9c3ab95e4c0))
* **gateway:** enforce remote-forbid with verified worker dispatch ([#546](https://github.com/superlinked/sie/issues/546)) ([19be847](https://github.com/superlinked/sie/commit/19be847b309803ca7c8b7a012670ea946d2ed66b))
* **gateway:** give a 503 serving refusal its own admission outcome ([#493](https://github.com/superlinked/sie/issues/493)) ([39a1aa9](https://github.com/superlinked/sie/commit/39a1aa95289b35f98b2eff2c07f8dd944b254b23))
* **gateway:** let a model access policy admit remote routes ([#591](https://github.com/superlinked/sie/issues/591)) ([7e7050f](https://github.com/superlinked/sie/commit/7e7050fd94a73425071f0a65b320dac8c9b3317a))
* **gateway:** opt in to pre-acceptance remote spill ([#550](https://github.com/superlinked/sie/issues/550)) ([4f3a62e](https://github.com/superlinked/sie/commit/4f3a62efbc7ea05023afa8adf37815447dc9f5d1))
* **gateway:** preserve validated remote routing policies ([#542](https://github.com/superlinked/sie/issues/542)) ([f05efe5](https://github.com/superlinked/sie/commit/f05efe5d994eef41445e120a58fdaf813a31b1a9))
* **gateway:** publish load-only model readiness work ([#541](https://github.com/superlinked/sie/issues/541)) ([bd945d6](https://github.com/superlinked/sie/commit/bd945d6c4ce7c4322e239d94b4fbb55a70debe57))
* **gateway:** restore cluster stream refusals before output ([#548](https://github.com/superlinked/sie/issues/548)) ([0ebc55c](https://github.com/superlinked/sie/commit/0ebc55c087bac5fa1f53b484feec691c415f9911))
* **helm:** add a remote worker pool that alone holds upstream credentials ([#496](https://github.com/superlinked/sie/issues/496)) ([4747581](https://github.com/superlinked/sie/commit/474758105b9f6f9dc265a7b4c4d7bdc3175e19b4))
* **helm:** give sie-config the upstream names the remote lanes define ([#508](https://github.com/superlinked/sie/issues/508)) ([72b48d6](https://github.com/superlinked/sie/commit/72b48d6ef15f2d1ba6555681ea75798fdfab4732))
* **helm:** limit the remote lanes' network access with a NetworkPolicy ([#498](https://github.com/superlinked/sie/issues/498)) ([f81fab3](https://github.com/superlinked/sie/commit/f81fab384a1cfbbb25a7d34ad8a0f3874e8807b3))
* **helm:** mount equivalence evidence into remote lanes ([#577](https://github.com/superlinked/sie/issues/577)) ([738cf38](https://github.com/superlinked/sie/commit/738cf38d7a3a39bfcb559c1620d1754aa9d82841))
* **models:** add a 1120 soft-token image profile for Gemma 4 31B on H100 ([#633](https://github.com/superlinked/sie/issues/633)) ([d97dcf2](https://github.com/superlinked/sie/commit/d97dcf24617493198063222d3fdbc55c03abc57d))
* **models:** add a compact 768-visual-token profile to tomoro-colqwen3-embed-4b ([#469](https://github.com/superlinked/sie/issues/469)) ([92a2a85](https://github.com/superlinked/sie/commit/92a2a85724becb3f1005f5eb5fe913bdb81601e6))
* **models:** add an 8,192-token output Gemma 4 31B hi-res profile ([#635](https://github.com/superlinked/sie/issues/635)) ([b9231ce](https://github.com/superlinked/sie/commit/b9231ced90b35bf0e813ee0d8b23b3a2a561aa08))
* **models:** add Qdrant/splade-ecommerce-esci sparse model ([#631](https://github.com/superlinked/sie/issues/631)) ([842c3b0](https://github.com/superlinked/sie/commit/842c3b0aca7c4e533a1ce225f78e888a03d4ecf5))
* **models:** refresh Qwen3.5-122B candidate ([44047bd](https://github.com/superlinked/sie/commit/44047bd17710ea85ac34dba985a255a28fd80ac3))
* **models:** refresh Qwen3.5-122B candidate ([e72f50f](https://github.com/superlinked/sie/commit/e72f50f4cb160e96313fd782e7b7f352448a90ea))
* **queue:** support load-only work and reply to backend fallback refusals ([#525](https://github.com/superlinked/sie/issues/525)) ([1aa45ce](https://github.com/superlinked/sie/commit/1aa45ce2078e0f4bc589fd63be1ff124e8b09720))
* **remote:** admit exact fresh OpenAI equivalence evidence ([#535](https://github.com/superlinked/sie/issues/535)) ([c6f077f](https://github.com/superlinked/sie/commit/c6f077f58a31a94f72b0e83174adf0b576b00158))
* **remote:** admit fresh matching SIE profile identity ([#540](https://github.com/superlinked/sie/issues/540)) ([98b55ec](https://github.com/superlinked/sie/commit/98b55ec091fa279c5b5a3079eb9dc74e94e8ded9))
* **remote:** bind equivalence evidence to execution identity, not one process ([#572](https://github.com/superlinked/sie/issues/572)) ([21f7c0b](https://github.com/superlinked/sie/commit/21f7c0b7fd5071a1ebcf5c7e74735deab081f962)), closes [#415](https://github.com/superlinked/sie/issues/415)
* **remote:** expose bounded numerical process diagnostics ([#561](https://github.com/superlinked/sie/issues/561)) ([9f966be](https://github.com/superlinked/sie/commit/9f966be23a56554368d5d0370fae7af2a69bf77f))
* **remote:** expose diagnostic numerical process snapshots ([#557](https://github.com/superlinked/sie/issues/557)) ([6113be1](https://github.com/superlinked/sie/commit/6113be1a631cf8d0956fdd3a924fe3c01e1a8101))
* **remote:** measure equivalence through a cluster gateway ([#579](https://github.com/superlinked/sie/issues/579)) ([e964598](https://github.com/superlinked/sie/commit/e96459863c4f4a016acc68040c8d84d51481c11b))
* **remote:** measure the local serving envelope in the equivalence probe ([#588](https://github.com/superlinked/sie/issues/588)) ([43157ba](https://github.com/superlinked/sie/commit/43157baa32af1ebd4f8201ad74897c2e8b4ff219))
* **remote:** re-verify the numerical admission before an admitted item runs ([#586](https://github.com/superlinked/sie/issues/586)) ([66f5635](https://github.com/superlinked/sie/commit/66f5635cee783fd0546151dc9ec0704c76b571bf)), closes [#415](https://github.com/superlinked/sie/issues/415)
* **remote:** report remote admissions in worker health ([#578](https://github.com/superlinked/sie/issues/578)) ([a85c04f](https://github.com/superlinked/sie/commit/a85c04f182439be5130d15e3675e77ef3e42dae3))
* route remote profiles to a dedicated remote worker bundle ([#492](https://github.com/superlinked/sie/issues/492)) ([ab5d4a9](https://github.com/superlinked/sie/commit/ab5d4a9919b1f8a839bf33408313dbf174f53352))
* **routing:** enable shared threshold routing behind cluster opt-in ([#555](https://github.com/superlinked/sie/issues/555)) ([93d5660](https://github.com/superlinked/sie/commit/93d56609724e23657d948a8f34a018f320dd5ae2))
* **sdk:** accept origin-confined configured HTTP clients ([#539](https://github.com/superlinked/sie/issues/539)) ([dea0ca1](https://github.com/superlinked/sie/commit/dea0ca1f771a0ba94729f46c5425da160fffbb0f))
* **sdk:** forbid remote serving and report which side served ([#495](https://github.com/superlinked/sie/issues/495)) ([17ce8b4](https://github.com/superlinked/sie/commit/17ce8b48bab955d351627e1664bcdfa4e9a2bf22))
* **sdk:** pass a target language to recommend ([#611](https://github.com/superlinked/sie/issues/611)) ([47cf609](https://github.com/superlinked/sie/commit/47cf60918ac3ed24f1d16a599706e243e73ba1fe))
* **server:** add a routing block to the model config ([#490](https://github.com/superlinked/sie/issues/490)) ([c02785e](https://github.com/superlinked/sie/commit/c02785e6ed703e4424b8621bdff7bbab22fe3728))
* **server:** add native GLiNER2.5 and Privacy Filter extraction ([#594](https://github.com/superlinked/sie/issues/594)) ([e01ad1e](https://github.com/superlinked/sie/commit/e01ad1e10e551d9f355947fcd9adf978e1a60a25))
* **server:** add native ZeRank 2 scoring ([#596](https://github.com/superlinked/sie/issues/596)) ([be8a688](https://github.com/superlinked/sie/commit/be8a688374d6c097096c93dee1bea814d653061d))
* **server:** add optional cuDNN SDPA startup control ([#650](https://github.com/superlinked/sie/issues/650)) ([8770c72](https://github.com/superlinked/sie/commit/8770c726d96ee8c0a39c1a484d41a9fc530cec1d))
* **server:** add Parakeet-TDT speech-to-text (nvidia/parakeet-tdt-0.6b-v3) ([#608](https://github.com/superlinked/sie/issues/608)) ([7e2a1d3](https://github.com/superlinked/sie/commit/7e2a1d31e48b3cea7560e06b9c40e2eaa3e1aa41))
* **server:** add Qwen3Guard-Gen-4B and Qwen3Guard-Gen-0.6B guard models ([#448](https://github.com/superlinked/sie/issues/448)) ([471189c](https://github.com/superlinked/sie/commit/471189c55c95160378d04d4254cdbd8ead9969d5))
* **server:** add SAM 3 open-vocabulary detection ([#614](https://github.com/superlinked/sie/issues/614)) ([7e7d0f5](https://github.com/superlinked/sie/commit/7e7d0f5adee80d8cc0fad1727d7723ba05c813d3))
* **server:** answer an upstream that cannot serve yet with a retryable 503 ([#494](https://github.com/superlinked/sie/issues/494)) ([ab0217b](https://github.com/superlinked/sie/commit/ab0217b4dbf2ec697eab4f9c29a3d231336737e3))
* **server:** bridge cold single-node generation before output ([#533](https://github.com/superlinked/sie/issues/533)) ([bab928f](https://github.com/superlinked/sie/commit/bab928fd1ba4465b65cd286c5b8ac24507695e35))
* **server:** cap the calls to each upstream and stop calling one that keeps failing ([#506](https://github.com/superlinked/sie/issues/506)) ([1a259cc](https://github.com/superlinked/sie/commit/1a259cc8f05f3a81b27c01b57a19165731ae35cb))
* **server:** collect process-bound numerical fleet evidence ([#556](https://github.com/superlinked/sie/issues/556)) ([7d0f487](https://github.com/superlinked/sie/commit/7d0f48704029ae2df7259a1257ffe3269e85ec99))
* **server:** count hybrid upstream generation with the model's tokenizer ([#620](https://github.com/superlinked/sie/issues/620)) ([2f9c107](https://github.com/superlinked/sie/commit/2f9c107309e8d1d0362aab88d3ef6f5b1c867e03))
* **server:** declare openai upstream endpoints and operator request fields ([#499](https://github.com/superlinked/sie/issues/499)) ([2292823](https://github.com/superlinked/sie/commit/2292823f4139bccb4372adade03b83a48211f921))
* **server:** default gliner-biomed-large-v1.0 to threshold 0.8, chosen on dev data ([#461](https://github.com/superlinked/sie/issues/461)) ([6c13f28](https://github.com/superlinked/sie/commit/6c13f2890f6af3d637770b94d8a86cbc5a3fcac6))
* **server:** disclose which side served and report routing in /v1/models ([#445](https://github.com/superlinked/sie/issues/445)) ([946b6fb](https://github.com/superlinked/sie/commit/946b6fb3e8f8d5e61d47745f06a4662dd2d382df))
* **server:** expose conservative immutable profile identity ([#530](https://github.com/superlinked/sie/issues/530)) ([1c62676](https://github.com/superlinked/sie/commit/1c626766a084e5562cbea0e50b0a98d10ef17f5f))
* **server:** identify flash BGE-M3 profiles with exact library builds ([#575](https://github.com/superlinked/sie/issues/575)) ([de82bad](https://github.com/superlinked/sie/commit/de82badfdc0c701602c909f2caf70c96010b7ffb)), closes [#415](https://github.com/superlinked/sie/issues/415)
* **server:** identify the BERT, cross-encoder and Nomic flash adapters ([#581](https://github.com/superlinked/sie/issues/581)) ([cd3cb91](https://github.com/superlinked/sie/commit/cd3cb9119438e251adce4fba35b7ed7bb341c9d5))
* **server:** load a remote profile without waiting for a local load ([#491](https://github.com/superlinked/sie/issues/491)) ([f53883b](https://github.com/superlinked/sie/commit/f53883ba543375748e015770a9bf42bc3a10be43))
* **server:** measure remote equivalence against local noise ([#531](https://github.com/superlinked/sie/issues/531)) ([e5cc5c6](https://github.com/superlinked/sie/commit/e5cc5c6504eab6b64cb73bb46aed6f02929c2477))
* **server:** preserve onboarded templates for direct remote chat ([#537](https://github.com/superlinked/sie/issues/537)) ([5581a23](https://github.com/superlinked/sie/commit/5581a234de068cd77d28b32d633bcb7db68dbc66))
* **server:** run the remote bundle on transformers 5 so hybrid counting loads transformers-5 tokenizers ([#625](https://github.com/superlinked/sie/issues/625)) ([577f3cf](https://github.com/superlinked/sie/commit/577f3cfa295a9ef1706eaabf1102fdd33ddbb4e1))
* **server:** serve a cold model through its remote profile on a single node ([#503](https://github.com/superlinked/sie/issues/503)) ([3742178](https://github.com/superlinked/sie/commit/374217831db87e94b7cef0d3b54c9ccbdaa43a9c))
* **server:** serve a remote-backed embedding model through an SIE upstream ([#437](https://github.com/superlinked/sie/issues/437)) ([7a56b3c](https://github.com/superlinked/sie/commit/7a56b3c6bbdeb2408f9a4392a567175acd16c93e))
* **server:** serve embeddings and rerank through an openai upstream ([#505](https://github.com/superlinked/sie/issues/505)) ([23cc6df](https://github.com/superlinked/sie/commit/23cc6df2eb6f86d23846459d7fcfa1dc4cec773c))
* **server:** serve native SIE chat through remote profiles ([#523](https://github.com/superlinked/sie/issues/523)) ([d0869f4](https://github.com/superlinked/sie/commit/d0869f443159a9914f009de71bbfeacc964a9309))
* **server:** serve OpenAI upstream chat and raw generation ([#528](https://github.com/superlinked/sie/issues/528)) ([21e2ab3](https://github.com/superlinked/sie/commit/21e2ab316eb82aa6f9b9bbd13087538395b8632d))
* **server:** serve remote chat through queued generation ([#529](https://github.com/superlinked/sie/issues/529)) ([a8bbfa4](https://github.com/superlinked/sie/commit/a8bbfa473b940f627374f4aa54810271fcd39454))
* **server:** serve score, extract and every encode output through an SIE upstream ([#501](https://github.com/superlinked/sie/issues/501)) ([6344850](https://github.com/superlinked/sie/commit/6344850a54ac76fb464b7b2e04ef9c4508245f52))
* **server:** SMVE sparse multi-vector encoding, with smve profiles for TopK models ([#518](https://github.com/superlinked/sie/issues/518)) ([9eeff51](https://github.com/superlinked/sie/commit/9eeff51a9074db7a1fe28a8fdb8b9759a5a675c3))
* **server:** stream generation through a remote SIE upstream ([#514](https://github.com/superlinked/sie/issues/514)) ([d3fa3c1](https://github.com/superlinked/sie/commit/d3fa3c16aa8150747da1fa3b2cd0624ba48a3ebd))
* **server:** support native GLiNER2 entity descriptions ([#605](https://github.com/superlinked/sie/issues/605)) ([461f8dc](https://github.com/superlinked/sie/commit/461f8dc5131b885d72d272406b78abd5d3f26fc4))
* **server:** support native LightOnOCR-3 extraction ([#632](https://github.com/superlinked/sie/issues/632)) ([5e60012](https://github.com/superlinked/sie/commit/5e6001278942d0c65420a0afe53a936947bc4eb5))
* **server:** validate shared remote chat responses ([#513](https://github.com/superlinked/sie/issues/513)) ([42897e9](https://github.com/superlinked/sie/commit/42897e9b4994f86add30173bffe43f0f01b26726))
* **telemetry:** observe cluster remote fallback persistence ([#551](https://github.com/superlinked/sie/issues/551)) ([8e2032b](https://github.com/superlinked/sie/commit/8e2032b3e7018a6926c5a32a38f2ebd3d3e870aa))
* **worker:** add versioned execution authority IPC methods ([#544](https://github.com/superlinked/sie/issues/544)) ([5f832a4](https://github.com/superlinked/sie/commit/5f832a4fab9764b5902a5efa5840670c1c42a0b5))
* **worker:** fence verified dispatch with versioned queues ([#545](https://github.com/superlinked/sie/issues/545)) ([2ba3d6e](https://github.com/superlinked/sie/commit/2ba3d6e3d02894cf4585c6bfb83b8513a1a8c928))


### Bug Fixes

* answer /v1/rerank without usage when the score has none ([#569](https://github.com/superlinked/sie/issues/569)) ([3459946](https://github.com/superlinked/sie/commit/3459946f282c61b73f6b0e1e1f5a0b25a317121b))
* **batcher:** serve long-form audio in its own lane, alternating with clips ([#601](https://github.com/superlinked/sie/issues/601)) ([45b6ffe](https://github.com/superlinked/sie/commit/45b6ffe1b58dd660a17ad0e2e5f9f9f174935b68)), closes [#585](https://github.com/superlinked/sie/issues/585)
* **examples:** align graph evidence with recorded calls ([#510](https://github.com/superlinked/sie/issues/510)) ([ef90dd6](https://github.com/superlinked/sie/commit/ef90dd602978cd7b02fe8820473e83e0f0f88191))
* **examples:** image-search scores SigLIP so400m-384, the model superlinked.com/image-search sells ([#476](https://github.com/superlinked/sie/issues/476)) ([395366c](https://github.com/superlinked/sie/commit/395366cd58d0c14eddfd59c84daec438acb5d8c8))
* **examples:** keep journal replay outside latency timing ([10564f8](https://github.com/superlinked/sie/commit/10564f8896407b390f9323f82ddc60160d1b99f2))
* **examples:** redact credentials in reply keys ([b57c033](https://github.com/superlinked/sie/commit/b57c0333b4b5e2068c6bf1dfbf81cba533ffd8ad))
* **examples:** reject malformed latency reply members ([e068b1d](https://github.com/superlinked/sie/commit/e068b1d8fa779c91fe69776389b4dd95415b8b95))
* **examples:** rerun and grade document-field extraction with the measured recipe ([#622](https://github.com/superlinked/sie/issues/622)) ([557f856](https://github.com/superlinked/sie/commit/557f8560d702c9ee00fddfa4ba2291cc91c106c0))
* **examples:** retain owned replies in timed event capture ([ab247dc](https://github.com/superlinked/sie/commit/ab247dc2979b19317e934d08a543e78a13228b4a))
* **examples:** scope OCR quality claims and support self-hosting ([#483](https://github.com/superlinked/sie/issues/483)) ([6cccc04](https://github.com/superlinked/sie/commit/6cccc04de9c6b78d3217854ad2377de36c0ffbd2))
* **examples:** sync latency outputs and retain deadline reasons ([c4a9274](https://github.com/superlinked/sie/commit/c4a927498abab822acb947f586d1187769374f89))
* **examples:** validate latency trial execution evidence ([c4e94a3](https://github.com/superlinked/sie/commit/c4e94a3639673b19bf991658ec81bf80a2b7da4b))
* **gateway:** answer a queued generation INPUT_TOO_LONG with 400 ([#497](https://github.com/superlinked/sie/issues/497)) ([0cb086d](https://github.com/superlinked/sie/commit/0cb086ddca0427b935524bc6a692fd8cbd3b9bff))
* **gateway:** derive a failed bridge's fallback error as the single server does ([#580](https://github.com/superlinked/sie/issues/580)) ([70b3181](https://github.com/superlinked/sie/commit/70b318162b452c48b6cc3f805e9dc3113386ce44))
* **gateway:** dispatch a forbid request normally when its model has no remote route ([#564](https://github.com/superlinked/sie/issues/564)) ([2600ded](https://github.com/superlinked/sie/commit/2600ded6831918106bdaa94936b243782c81be04))
* **gateway:** keep a hidden remote profile out of the model listing ([#598](https://github.com/superlinked/sie/issues/598)) ([7a43250](https://github.com/superlinked/sie/commit/7a43250ddc1a5541fe1024a2ab11063979273f66))
* **gateway:** keep json_schema property order through to the worker ([#455](https://github.com/superlinked/sie/issues/455)) ([571f45b](https://github.com/superlinked/sie/commit/571f45bf5c49caf85d7780c4b224c2c5b5b33173))
* **gateway:** keep requests for undeclared outputs off numerical bridges ([#597](https://github.com/superlinked/sie/issues/597)) ([b6f1702](https://github.com/superlinked/sie/commit/b6f17028156b30096cade2bb36bdd8455ff681f1))
* **helm:** require pytorch for remote lanes ([2e1aef2](https://github.com/superlinked/sie/commit/2e1aef270914c849c0c93221286a0afce1bf61a9))
* **helm:** require pytorch for remote lanes ([1545d61](https://github.com/superlinked/sie/commit/1545d6118ce8a1fd2226a6efed7d358fb1c97bc8))
* match identifier patterns against the whole value ([#568](https://github.com/superlinked/sie/issues/568)) ([6e5ce3d](https://github.com/superlinked/sie/commit/6e5ce3d5dc6e11bfd48f06d11bf16be8d72c3ce2))
* **models:** pin GLiNER v2.5 tokenizer dependencies ([#634](https://github.com/superlinked/sie/issues/634)) ([a52cdbb](https://github.com/superlinked/sie/commit/a52cdbb0b5bd2a14042b441df6e6e5e867f2c944))
* **models:** raise the Hy-MT2-1.8B output cap to 1024 tokens ([#610](https://github.com/superlinked/sie/issues/610)) ([e831d4d](https://github.com/superlinked/sie/commit/e831d4d89d733453168f951208472f872c731a99))
* **models:** validate Qwen runtime defaults ([8fae464](https://github.com/superlinked/sie/commit/8fae464d16cdc138efffcd0f1a22e201a2bffa7b))
* **parakeet:** decode long audio in pieces cut at pauses ([#624](https://github.com/superlinked/sie/issues/624)) ([4d0e3c6](https://github.com/superlinked/sie/commit/4d0e3c6a0680f73db99fea1a9769562bbd367e6f))
* **queue:** answer a remote profile's upstream refusal at once with its wait ([#570](https://github.com/superlinked/sie/issues/570)) ([22aa825](https://github.com/superlinked/sie/commit/22aa8250d41a67cbb14077e27f75a1ce1dc3b57a))
* **remote:** bind profile identity to observed hardware and BLAS ([#536](https://github.com/superlinked/sie/issues/536)) ([89640b0](https://github.com/superlinked/sie/commit/89640b0f6ae2c9e90d96d1baf7462b60726e5121))
* **remote:** bound configuration waits and settle grammar refusals ([#558](https://github.com/superlinked/sie/issues/558)) ([b6bbf38](https://github.com/superlinked/sie/commit/b6bbf38d1e539e470430d573e71d16309bd0eabb))
* **remote:** carry admitted numerical work on its own subject ([#589](https://github.com/superlinked/sie/issues/589)) ([6d4d74b](https://github.com/superlinked/sie/commit/6d4d74b13fcbaa25df39a0df4e2895a51740517f))
* **remote:** keep instructions and unadmitted plans off numerical bridges ([#600](https://github.com/superlinked/sie/issues/600)) ([ab016b1](https://github.com/superlinked/sie/commit/ab016b13b4417baabf6964c745eedfda3b52ad0f))
* **remote:** name both identities when SIE identity admission refuses a mismatch ([#563](https://github.com/superlinked/sie/issues/563)) ([52c6c6d](https://github.com/superlinked/sie/commit/52c6c6d8d7843a412be9fc005f9f6b7ce9d2c50d))
* **sdk:** preserve HTTP retry hints in terminal server errors ([#560](https://github.com/superlinked/sie/issues/560)) ([7264193](https://github.com/superlinked/sie/commit/72641933c2d822ec2f38ea4c398bc08eeb3e3be3))
* **sdk:** serialize encode query roles in nested options ([#617](https://github.com/superlinked/sie/issues/617)) ([812962d](https://github.com/superlinked/sie/commit/812962d5a6ad22e7d5f08f75b1ab719789503ce4))
* **server:** apply default_instruction in the remaining flash embedders ([#623](https://github.com/superlinked/sie/issues/623)) ([07983db](https://github.com/superlinked/sie/commit/07983dbfe60812533468989318f7bcbaf726f5d1))
* **server:** apply default_instruction to qwen2_flash queries ([#621](https://github.com/superlinked/sie/issues/621)) ([db4c0a8](https://github.com/superlinked/sie/commit/db4c0a86d90e30b21a008dadfe9cd52fcbd043aa))
* **server:** classify all of a long text with GLiNER2, not only its first 512 words ([#449](https://github.com/superlinked/sie/issues/449)) ([82cbf2e](https://github.com/superlinked/sie/commit/82cbf2ec7f5c757e18e95eba50595c4d5f5806ef))
* **server:** convert queued tool-call arguments by the declared schema ([#609](https://github.com/superlinked/sie/issues/609)) ([5f00470](https://github.com/superlinked/sie/commit/5f0047091c907a92e43e3bfd764fd3d455c5793f))
* **server:** default grammar-constrained generation to greedy sampling ([#453](https://github.com/superlinked/sie/issues/453)) ([b1c8887](https://github.com/superlinked/sie/commit/b1c88878833fbb0367660302520e71657fbdfcd5))
* **server:** exit on SIGTERM after starting the drain ([#637](https://github.com/superlinked/sie/issues/637)) ([ecb957d](https://github.com/superlinked/sie/commit/ecb957db6f4503ac7e777be74c4b3a9a50e8a260))
* **server:** give background identity refreshes their own budget ([#618](https://github.com/superlinked/sie/issues/618)) ([c998453](https://github.com/superlinked/sie/commit/c99845360eab0f7f11dbe5a67966dda3de081ecf))
* **server:** honor explicitly selected default profiles ([#534](https://github.com/superlinked/sie/issues/534)) ([87a0759](https://github.com/superlinked/sie/commit/87a0759434d97a6f4316916652e49e71132c150b))
* **server:** ignore optional triton type imports ([10cac53](https://github.com/superlinked/sie/commit/10cac53729f760f0619f01cf440cbaf74e8fbab7))
* **server:** ignore optional triton type imports ([f9ab207](https://github.com/superlinked/sie/commit/f9ab2079389ba3f6f8df63e5714d7abfbb3a3e6f))
* **server:** let GLiNER checkpoints on byte-level BPE encoders read split words ([#460](https://github.com/superlinked/sie/issues/460)) ([d214552](https://github.com/superlinked/sie/commit/d214552a699dfd591ead7557e48157ea0bd42253))
* **server:** meter ColQwen3 text from processed inputs ([#509](https://github.com/superlinked/sie/issues/509)) ([d787725](https://github.com/superlinked/sie/commit/d787725f825b451e650799c8fb8c515fea46bbd2))
* **server:** pin GLiFormer decoding probe to float32 ([#562](https://github.com/superlinked/sie/issues/562)) ([df09d4b](https://github.com/superlinked/sie/commit/df09d4b7c94de438e58cc44fa0e325e12cac2288))
* **server:** read long documents whole in GLiNER extract ([#451](https://github.com/superlinked/sie/issues/451)) ([2f97d02](https://github.com/superlinked/sie/commit/2f97d022440f61403efea9badc97c55a16bb45ce))
* **server:** refuse LoRAs that train biases ([#583](https://github.com/superlinked/sie/issues/583)) ([dcfdf51](https://github.com/superlinked/sie/commit/dcfdf51e695be990f0abed5b006c8ec47ab544b9))
* **server:** refuse remote generation before streaming headers ([#532](https://github.com/superlinked/sie/issues/532)) ([9719ae1](https://github.com/superlinked/sie/commit/9719ae16413d59117de55bdd72d125f5ea7d0153))
* **server:** reject a config delta per model instead of stalling its bundle ([#489](https://github.com/superlinked/sie/issues/489)) ([81cc427](https://github.com/superlinked/sie/commit/81cc42789544bbc3700769771b7d409c4228b846))
* **server:** reject incomplete GLiNER2 extraction items ([#552](https://github.com/superlinked/sie/issues/552)) ([689e4d8](https://github.com/superlinked/sie/commit/689e4d808cbd66008500da8f58d8fa6e71510b8f))
* **server:** report GLiNER2 caller errors as invalid input ([#526](https://github.com/superlinked/sie/issues/526)) ([96066a3](https://github.com/superlinked/sie/commit/96066a311fd991ed32b57ece62fb92b09ae47ea1))
* **server:** report NLI zero-shot extract usage and enforce the extract label limit on the queue path ([#447](https://github.com/superlinked/sie/issues/447)) ([2cf3d89](https://github.com/superlinked/sie/commit/2cf3d89382955f1b2cf592662de93cb332303f47))
* **server:** score a lone GLiClass label on its own in single-label mode ([#616](https://github.com/superlinked/sie/issues/616)) ([45c3fee](https://github.com/superlinked/sie/commit/45c3fee8a4d649e987e93b0c899f947e10393b61))
* **server:** score NLI zero-shot labels like the transformers pipeline ([#615](https://github.com/superlinked/sie/issues/615)) ([d134ab2](https://github.com/superlinked/sie/commit/d134ab29815d092d7773b375a9cc5d7be180a3a0))
* **server:** score Qwen3 text rerankers in float32 so top candidates stop tying ([#440](https://github.com/superlinked/sie/issues/440)) ([8de63e3](https://github.com/superlinked/sie/commit/8de63e3f3e7719b4152fe7babfb8a1733277fd4f))
* **server:** size NLI zero-shot extract batches by the rows each item runs ([#465](https://github.com/superlinked/sie/issues/465)) ([57a9856](https://github.com/superlinked/sie/commit/57a98562cba9d1b9f54fab79f14e5654f35a2402))
* **server:** skip remote profiles a worker refuses in pool isolation ([#627](https://github.com/superlinked/sie/issues/627)) ([cb79fc1](https://github.com/superlinked/sie/commit/cb79fc1eafff8d7ffc51d6ade190b05c4422c0ef))
* **server:** suppress private chat reasoning logprobs independently of thinking policy ([#527](https://github.com/superlinked/sie/issues/527)) ([d9d63dc](https://github.com/superlinked/sie/commit/d9d63dcca3c900cb7a49bd832eca1c287587a5bf))
* **sglang:** bound JSON number digits in XGrammar json_schema grammars ([#467](https://github.com/superlinked/sie/issues/467)) ([7f3666d](https://github.com/superlinked/sie/commit/7f3666dd785a215274eb2569f3a9aee9a4826a00))
* validate remote routing before configuration writes ([#512](https://github.com/superlinked/sie/issues/512)) ([cb95d4a](https://github.com/superlinked/sie/commit/cb95d4ac753492d5d854cdcc6a61ab995e207948))
* **whisper:** transcribe long-form audio one recording per pipeline call ([#582](https://github.com/superlinked/sie/issues/582)) ([8bdaeba](https://github.com/superlinked/sie/commit/8bdaebac7e7480c19d5254e8b5161f5293e9a745)), closes [#574](https://github.com/superlinked/sie/issues/574)
* **worker:** enforce live configuration authority through execution ([#543](https://github.com/superlinked/sie/issues/543)) ([f6c530c](https://github.com/superlinked/sie/commit/f6c530cf9cce813d052deca316b4a1e439514f77))


### Performance Improvements

* **colbert:** tokenize a ModernColBERT batch in one call and pack it on the host ([#646](https://github.com/superlinked/sie/issues/646)) ([e15d014](https://github.com/superlinked/sie/commit/e15d0149d611e1a5f5f1d877ddac57ebecbbcddf))
* **gliner:** run 32 rows per forward pass instead of gliner's default 8 ([#638](https://github.com/superlinked/sie/issues/638)) ([e7622c3](https://github.com/superlinked/sie/commit/e7622c3d314e8c104fc73dccf40c728ab9ac9718))
* **gliner:** turn the adaptive batching controller off on the GLiNER profiles ([#648](https://github.com/superlinked/sie/issues/648)) ([cfaafe3](https://github.com/superlinked/sie/commit/cfaafe356bca2f7c320e53aa834651e48bea04f8))
* **models:** add a 16-request CUDA-graph H100 profile for Qwen3.8-27B-FP8 ([#454](https://github.com/superlinked/sie/issues/454)) ([409fee6](https://github.com/superlinked/sie/commit/409fee6709ed1b55cc9cdcfa886d6c7a37a88cac))
* **models:** compile compact JSON grammars on every Qwen3.6-35B-A3B profile ([#636](https://github.com/superlinked/sie/issues/636)) ([d8f58cb](https://github.com/superlinked/sie/commit/d8f58cb85dbe69564b2ba0ebf8718dd7dd955eba))
* **models:** compile compact, digit-bounded JSON grammars on the Gemma 4 31B hi-res profiles ([#641](https://github.com/superlinked/sie/issues/641)) ([a8eb93d](https://github.com/superlinked/sie/commit/a8eb93d9d7ee2a910f3256e6486454d897a81f24))
* **models:** enable decode CUDA graphs on the Qwen3.8-27B-FP8 bare route and overlap the batch lane ([#640](https://github.com/superlinked/sie/issues/640)) ([0f0d691](https://github.com/superlinked/sie/commit/0f0d691beae91d6177b660e3b6d68f67f7383556))
* **models:** read a single Qwen3.8-27B-FP8 page image at 3.2 megapixels and compile compact JSON grammars ([#466](https://github.com/superlinked/sie/issues/466)) ([329df87](https://github.com/superlinked/sie/commit/329df878acd18ffdd5ca3decba5605a619c16840))
* **models:** serve Qwen3.6-35B-A3B default on the CUDA 13 engine with CUDA graphs ([#639](https://github.com/superlinked/sie/issues/639)) ([2f7fcbd](https://github.com/superlinked/sie/commit/2f7fcbda0e2d86703bcfe5a844d580e7992ee571))
* **qwen3-embedding:** fuse the 8B's elementwise layer work into exact Triton kernels ([#645](https://github.com/superlinked/sie/issues/645)) ([6949a8b](https://github.com/superlinked/sie/commit/6949a8b935c1f29950b4dd6b0f333cc224298e54))
* **qwen3-embedding:** keep up to four Qwen3-Embedding-4B batches in flight to SGLang ([#647](https://github.com/superlinked/sie/issues/647)) ([99140a4](https://github.com/superlinked/sie/commit/99140a475308de3199e3aba8302b3fcfa7fa3b00))
* **sam3:** skip the unused mask decoder, encode images one at a time and cache label encodings ([#643](https://github.com/superlinked/sie/issues/643)) ([06bc893](https://github.com/superlinked/sie/commit/06bc893a256775d694cbcc4c38cb54fc20f1ac40))
* **server:** run SGLang OCR batches concurrently instead of one at a time ([#462](https://github.com/superlinked/sie/issues/462)) ([dce7611](https://github.com/superlinked/sie/commit/dce7611654a3bd25b76405dbdb0415c8ec92cd40))
* **server:** serve OWLv2 with the fast image processor and shrink oversized photos ([#473](https://github.com/superlinked/sie/issues/473)) ([b01062d](https://github.com/superlinked/sie/commit/b01062d8c3566dfb27d6e00ce9291d713a842d43))
* **sparse:** pool before the sparse activation and keep doc-v3 batches at full size ([#642](https://github.com/superlinked/sie/issues/642)) ([9075c1f](https://github.com/superlinked/sie/commit/9075c1f6373f0f728ebb205b4f94d339ec8dfb09))

## [0.9.0](https://github.com/superlinked/sie/compare/v0.8.3...v0.9.0) (2026-09-30)


### ⚠ BREAKING CHANGES

* **helm:** nats.auth.enabled defaults to true. Upgrading restarts NATS and rolls sie-config, the gateway, and the workers; pods that have not rolled yet are refused until they do, and memory-backed queued work is lost as on any NATS restart. To avoid the gap, upgrade once with nats.auth.allowAnonymous=true, then again without it. helm upgrade --reuse-values now fails the render because the reused NATS values lack the server wiring; use --reset-then-reuse-values or -f. With an external NATS server (nats.install=false), set nats.auth.existingSecrets.{config,gateway,worker} and create the users, or set nats.auth.enabled=false. nats-box and the NATS helm test pod are disabled by default. Credentials embedded in SIE_NATS_URL are not used.
* **helm:** values-aws.yaml, values-gke.yaml, and values-aks.yaml no longer enable the gateway Ingress, so a plain upgrade with them removes the existing host-less, TLS-less Ingress. A gateway Ingress now renders only with gateway auth (gateway.auth.mode=static with gateway.auth.tokenSecretName), the oauth2-proxy edge on ingress-nginx, or ingress.allowUnauthenticated=true, and only with TLS or ingress.allowPlaintext=true; set both opt-ins to keep the previous catch-all Ingress. A LoadBalancer or NodePort gateway Service needs gateway auth or gateway.service.allowUnauthenticated=true. POST /v1/pools now rejects a warm floor above SIE_GATEWAY_POOL_MAX_MINIMUM_WORKER_COUNT (default 4) or a TTL above SIE_GATEWAY_POOL_MAX_TTL_S (default 3600) with 400, a pool beyond SIE_GATEWAY_MAX_POOLS (default 64) or named default with 403. The Python and TypeScript SDKs now send SIE_API_KEY when no API key is passed and the base URL has the same origin as SIE_BASE_URL; pass an empty API key to opt out.
* **config:** the gateway no longer receives the sie-config admin token, so with gateway auth enabled its admin routes (POST, PUT and DELETE under /v1/pools, /v1/admin and /v1/configs) answer 403 until gateway.auth.adminTokenSecretName is set. sie-config, gateway and worker sidecar images older than this chart do not work with the split tokens.

### Features

* **examples:** add document-to-markdown-olmocr, LightOnOCR-2-1B on all of olmOCR-Bench ([#441](https://github.com/superlinked/sie/issues/441)) ([934a2a6](https://github.com/superlinked/sie/commit/934a2a6fdf4d36b41d8bfb8aaf001ec869e5ed31))
* **examples:** add support-assistant-policy, the /chat support-rules run ([#438](https://github.com/superlinked/sie/issues/438)) ([533ed0f](https://github.com/superlinked/sie/commit/533ed0f28da12f5648a272df616131b42641bd57))
* **generation:** report prefix-cache hits as usage.prompt_tokens_details.cached_tokens ([#386](https://github.com/superlinked/sie/issues/386)) ([c125504](https://github.com/superlinked/sie/commit/c12550414a1da8b78e4b883487f1f42c7b0dc6f9))
* **helm:** authenticate every NATS connection with per-component users ([#421](https://github.com/superlinked/sie/issues/421)) ([60f30ab](https://github.com/superlinked/sie/commit/60f30ab4a6f11c95a65d0f016e7c3840bd08e823))
* **models:** add tencent/Hy-MT2-1.8B ([#360](https://github.com/superlinked/sie/issues/360)) ([08c111f](https://github.com/superlinked/sie/commit/08c111ff91dad7ea6a4f070396155d4f06270f79))
* **models:** add thinking profiles for Qwen3.8-27B-FP8 ([#383](https://github.com/superlinked/sie/issues/383)) ([ad87d96](https://github.com/superlinked/sie/commit/ad87d963f57a738df7ff37da9e8f69ed8b5e8bc3))
* **score:** report the caller's content tokens separately from the reranker prompt template ([#442](https://github.com/superlinked/sie/issues/442)) ([b17a906](https://github.com/superlinked/sie/commit/b17a90664588fd48c2f8f50168b32e60c314da78))
* **server:** add operator-defined upstreams with credential and egress controls ([#431](https://github.com/superlinked/sie/issues/431)) ([fe22241](https://github.com/superlinked/sie/commit/fe22241111ea334d9caa4aa22040c817472db4d4))
* **server:** serve GLiClass multilang-ultra and the layer-wise v1.0 checkpoints ([#420](https://github.com/superlinked/sie/issues/420)) ([5699f3e](https://github.com/superlinked/sie/commit/5699f3eedc35780891f836dc5c791f6caa388957))
* **server:** serve TopK-Embed-V1 multi-vector models (0.8B and 2B) ([#417](https://github.com/superlinked/sie/issues/417)) ([36c6129](https://github.com/superlinked/sie/commit/36c612915a4ce51c1baa2207cde0663b76744502))


### Bug Fixes

* align deadline dashboards and workspace resolver contracts ([#412](https://github.com/superlinked/sie/issues/412)) ([a0805c3](https://github.com/superlinked/sie/commit/a0805c3fdc5aa3c40bb3cd721931b010076f8da6))
* **config:** give gateways and worker sidecars a read-only sie-config token ([#418](https://github.com/superlinked/sie/issues/418)) ([ef3eb69](https://github.com/superlinked/sie/commit/ef3eb69b331432cdb0411d8d643441508e2eda24))
* **config:** validate model config writes against the worker schema and support chart rollback ([#392](https://github.com/superlinked/sie/issues/392)) ([296998f](https://github.com/superlinked/sie/commit/296998f630f252a991a30e12c4fc048482afb6f8))
* **gateway:** carry request deadlines to workers and bound direct generation ([#401](https://github.com/superlinked/sie/issues/401)) ([8ec3b7e](https://github.com/superlinked/sie/commit/8ec3b7ec4245b95540fc2d0a72df1110d5b475ad))
* **gateway:** distinguish request body read failures from size limits ([#385](https://github.com/superlinked/sie/issues/385)) ([d7e0ffa](https://github.com/superlinked/sie/commit/d7e0ffac9116864127d865e03249802ed4e95fba))
* **gateway:** fence configuration exports against concurrent updates ([#387](https://github.com/superlinked/sie/issues/387)) ([1b83d3e](https://github.com/superlinked/sie/commit/1b83d3e06cf4762cf27c006bc9bef51dab1673bb))
* **gateway:** keep other bundle models routable while workers predate a new adapter ([#419](https://github.com/superlinked/sie/issues/419)) ([a6eb6bb](https://github.com/superlinked/sie/commit/a6eb6bbcf02d1a43772a0d12cc2b5dd306f9795c))
* **generation:** verify strict structured output and serve grammars on grammar-safe profiles ([#399](https://github.com/superlinked/sie/issues/399)) ([5e2ae50](https://github.com/superlinked/sie/commit/5e2ae504b031f6d4182c5e731fbb115881a9e110))
* **helm:** generate the sie-config admin token and gate gateway readiness ([#396](https://github.com/superlinked/sie/issues/396)) ([7060f20](https://github.com/superlinked/sie/commit/7060f20c6e3a3df727800ba3caa1101dc61c0493))
* **helm:** require authenticated, TLS-protected gateway exposure and bound the pool API ([#393](https://github.com/superlinked/sie/issues/393)) ([bc2e66a](https://github.com/superlinked/sie/commit/bc2e66a2aad6083dd74a3a862d32a7cc15e4f5bb))
* **integrations:** send explicit item ids so rerankers map scores back ([#410](https://github.com/superlinked/sie/issues/410)) ([c07db06](https://github.com/superlinked/sie/commit/c07db0638d51b68e86154682660ec43320421efe))
* keep out-of-range timeouts from panicking the gateway and sidecar ([#411](https://github.com/superlinked/sie/issues/411)) ([658d1d4](https://github.com/superlinked/sie/commit/658d1d464618d8aefd88fee54b53f0b4ef31f24a))
* **models:** serve Iso-ModernColBERT with its PyLate recipe ([#436](https://github.com/superlinked/sie/issues/436)) ([422232f](https://github.com/superlinked/sie/commit/422232f178463719ea8bb97cf6465c75feb13c6b))
* preserve trace context across local ingest handoffs ([#405](https://github.com/superlinked/sie/issues/405)) ([1e12c2d](https://github.com/superlinked/sie/commit/1e12c2dd579ca556f02e5e91d675182deb5f9776))
* **runtime:** preserve model retries and parked shutdown settlement ([#408](https://github.com/superlinked/sie/issues/408)) ([7d70326](https://github.com/superlinked/sie/commit/7d70326d287e5cc4121030c88a2c513d7e91d6c1))
* **runtime:** recover MLX exits and isolate invalid model configs ([#413](https://github.com/superlinked/sie/issues/413)) ([51699dc](https://github.com/superlinked/sie/commit/51699dcf8683b547726b8d477466fa87e9c8cb6b))
* **sdk:** retry only requests that never reached the server ([#400](https://github.com/superlinked/sie/issues/400)) ([0860d41](https://github.com/superlinked/sie/commit/0860d417ed6097eea4a608fdda6fb2b2b7a405eb))
* **server:** fail only the over-long GLiClass item under overflow_policy error ([#429](https://github.com/superlinked/sie/issues/429)) ([0db8a34](https://github.com/superlinked/sie/commit/0db8a34936f27fab30cb5f2e15abc765dc6782ba))
* **server:** keep loaded models serving while another model loads or is evicted ([#398](https://github.com/superlinked/sie/issues/398)) ([88ee59a](https://github.com/superlinked/sie/commit/88ee59aac10006d1aa658bdf256fa03abf959d49))
* **server:** pause GLiClass graph recording after an out-of-memory first replay ([#434](https://github.com/superlinked/sie/issues/434)) ([1f5426d](https://github.com/superlinked/sie/commit/1f5426d72d95c8f5ab0633d1e3dba2563bfe58c1))
* **server:** refuse GLiClass items whose labels leave no room for the document ([#427](https://github.com/superlinked/sie/issues/427)) ([b5778bd](https://github.com/superlinked/sie/commit/b5778bd2fcb3facd7cecad2d27f0fdb3fad0d597))
* **server:** reject token ids and apply model profiles on /v1/embeddings ([#395](https://github.com/superlinked/sie/issues/395)) ([078c6e5](https://github.com/superlinked/sie/commit/078c6e5d3f1b7b28544d8434ab600ab60106837a))
* **server:** retry transient model-load failures and reload exited engines ([#394](https://github.com/superlinked/sie/issues/394)) ([8767c81](https://github.com/superlinked/sie/commit/8767c819e41cce74145a5ca4b96a31a0ffe3ae32))
* **server:** send float16 multivectors to the sidecar as bytes ([#416](https://github.com/superlinked/sie/issues/416)) ([7ee9f75](https://github.com/superlinked/sie/commit/7ee9f75ef54abdf151ab125615ca015b03168d83))
* **telemetry:** count sidecar barrier NAKs of unsupported models as model_unsupported ([#432](https://github.com/superlinked/sie/issues/432)) ([a009826](https://github.com/superlinked/sie/commit/a009826ccaf8dc6e57c1c7403efa411eb8502426))
* **telemetry:** preserve long finite lifecycle durations ([#407](https://github.com/superlinked/sie/issues/407)) ([0edcfb3](https://github.com/superlinked/sie/commit/0edcfb373eaa543162265e730617e87b8a7ee930))
* **telemetry:** preserve safe per-request batch timing ([#402](https://github.com/superlinked/sie/issues/402)) ([f0cb865](https://github.com/superlinked/sie/commit/f0cb8652084390d0775a9315e8a1b41b1141e9af))
* **telemetry:** raise the sidecar NATS series budget for the deadline reason ([#428](https://github.com/superlinked/sie/issues/428)) ([db02e21](https://github.com/superlinked/sie/commit/db02e2196760129607ec25265a2303499ae0f2af))
* **telemetry:** rebuild remote metric scalar attributes ([#406](https://github.com/superlinked/sie/issues/406)) ([91079ac](https://github.com/superlinked/sie/commit/91079ac30c45f7b57e085b5af60b165d74847e5e))
* **telemetry:** record failures and streamed response lifecycle ([#404](https://github.com/superlinked/sie/issues/404)) ([e4e7ec0](https://github.com/superlinked/sie/commit/e4e7ec0b4a422329c43b098bf91703315a20f940))
* **telemetry:** validate exported log resource and completion fields ([#409](https://github.com/superlinked/sie/issues/409)) ([ed0ad32](https://github.com/superlinked/sie/commit/ed0ad32c0eb6d1c75f6eae36940b40331c269da9))


### Performance Improvements

* **models:** load five more DeBERTa-v3 GLiClass models with bucketed CUDA graphs ([#422](https://github.com/superlinked/sie/issues/422)) ([aa9b323](https://github.com/superlinked/sie/commit/aa9b3230142fe7691e5e9325c77e2d3d1f807581))
* **models:** raise GLiClass windows to the reference 1,024 tokens ([#425](https://github.com/superlinked/sie/issues/425)) ([7c89f6b](https://github.com/superlinked/sie/commit/7c89f6ba26a56d95031bb13e4a86425fa5b0074b))
* **models:** serve MADLAD from its bfloat16 CTranslate2 artifact ([#433](https://github.com/superlinked/sie/issues/433)) ([7256a68](https://github.com/superlinked/sie/commit/7256a687e3bca40b350f81928eed51bb6ad15088))
* **server:** bound GLiClass CUDA graph keys so mixed traffic keeps replaying ([#389](https://github.com/superlinked/sie/issues/389)) ([5c50c33](https://github.com/superlinked/sie/commit/5c50c33089cf6b38bb0b8414895c3206647ff706))
* **server:** fuse the ModernBERT flash RoPE for models as accurate against float32 ([#424](https://github.com/superlinked/sie/issues/424)) ([f18d6d5](https://github.com/superlinked/sie/commit/f18d6d5e1abf62d01ee1e929d31050907a4f5022))
* **server:** replay ModernBERT flash forwards as CUDA graphs ([#423](https://github.com/superlinked/sie/issues/423)) ([4c4c3a2](https://github.com/superlinked/sie/commit/4c4c3a21eb079b56aea7623c2bc50cab80b4c3d2))
* **server:** run GLiClass ModernBERT encoders through the flash-attention varlen stack ([#414](https://github.com/superlinked/sie/issues/414)) ([77577f9](https://github.com/superlinked/sie/commit/77577f969ed6ed659a1ba0c0db78e1aa73ba11b9))

## [0.8.3](https://github.com/superlinked/sie/compare/v0.8.2...v0.8.3) (2026-09-26)


### Features

* **examples:** add a typed-decisions verification example ([#362](https://github.com/superlinked/sie/issues/362)) ([d4e00b3](https://github.com/superlinked/sie/commit/d4e00b3b33701e150d28fe191df4a50d2bc96416))
* **examples:** add watermark robustness eval (translation vs paraphrase) ([#296](https://github.com/superlinked/sie/issues/296)) ([a7c37ac](https://github.com/superlinked/sie/commit/a7c37ace008d4576a96e2085780441fa48249796))
* **examples:** refresh typed-decisions evidence on a faster server and add GLiNER2.5-Decide ([#376](https://github.com/superlinked/sie/issues/376)) ([8b4dc45](https://github.com/superlinked/sie/commit/8b4dc459dea70ca9bf9ff2320a459663b88ba2a3))
* **server:** add GLiFormer multi-task extraction models ([#358](https://github.com/superlinked/sie/issues/358)) ([28a07f5](https://github.com/superlinked/sie/commit/28a07f500490c8bd2653b68f83baf3718cb70b95))
* **server:** encode GLiClass label groups separately by default ([#364](https://github.com/superlinked/sie/issues/364)) ([71b395f](https://github.com/superlinked/sie/commit/71b395fee4564df7a33fdc47561427c2a5044c4e))
* **server:** GLiClass task prompts, few-shot examples and label groups, plus instruct, Opir and multilingual models ([#355](https://github.com/superlinked/sie/issues/355)) ([bc1d66e](https://github.com/superlinked/sie/commit/bc1d66ef3df2beb93763a0bf874cd41e0b221d89))
* **server:** relations from GLiNER relex models, plus GLiNER bi-encoder v2 and PII models ([#356](https://github.com/superlinked/sie/issues/356)) ([5914e91](https://github.com/superlinked/sie/commit/5914e9111443ba84999d559419a3be6fed23d236))
* **server:** replay GLiClass forwards as CUDA graphs ([#367](https://github.com/superlinked/sie/issues/367)) ([7afc169](https://github.com/superlinked/sie/commit/7afc1691422c386ba276a2277c81675efff34943))
* **server:** serve GLiNER2.5-Decide typed-decision models ([#368](https://github.com/superlinked/sie/issues/368)) ([2bbbc54](https://github.com/superlinked/sie/commit/2bbbc5400cc3926501fa1a706b12bf578e31c373))
* **server:** serve Laya typed-decision models through extract ([#353](https://github.com/superlinked/sie/issues/353)) ([5162aa7](https://github.com/superlinked/sie/commit/5162aa7d006e3c5fb0d8f26f967c049bb63ccd79))


### Bug Fixes

* **examples:** follow the screenshot-mining and multi-vector page selections ([#349](https://github.com/superlinked/sie/issues/349)) ([72c74bb](https://github.com/superlinked/sie/commit/72c74bb2135b9960cbc2bf20c46c361589484d36))
* **examples:** score /structured-output's amended run, and drop the last display claim ([#352](https://github.com/superlinked/sie/issues/352)) ([26c81c2](https://github.com/superlinked/sie/commit/26c81c2483dd59a6d91575fb759423a25d7d81f7))
* **examples:** score the detect and image-classify pages as they now publish ([#350](https://github.com/superlinked/sie/issues/350)) ([e48c1a6](https://github.com/superlinked/sie/commit/e48c1a64b000754d674a9f0d02090f5fece81ff4))
* **examples:** score the redact composition, and stop both examples describing their pages ([#357](https://github.com/superlinked/sie/issues/357)) ([e9815c3](https://github.com/superlinked/sie/commit/e9815c332f967cbc4b1ef2635ed9bd04024023b3))
* **examples:** stop chat and caption-vqa describing what their pages display ([#351](https://github.com/superlinked/sie/issues/351)) ([2e3c45e](https://github.com/superlinked/sie/commit/2e3c45ec8cdb7e6e7ea14ca6edc8e6b876ebd381))
* **examples:** stop guardrails and speech-to-text describing what their pages display ([#354](https://github.com/superlinked/sie/issues/354)) ([79ae97e](https://github.com/superlinked/sie/commit/79ae97ea007ef3bd07c1efdb700edfb5c80ac540))
* **gliformer:** bound span decoding and keep padding out of it ([#375](https://github.com/superlinked/sie/issues/375)) ([0ee26b0](https://github.com/superlinked/sie/commit/0ee26b0337c82c11a2792607e90c4cf702ca3a39))
* **knowledge-graph:** narrow two relation schemas and re-pin the evidence ([#348](https://github.com/superlinked/sie/issues/348)) ([72a9924](https://github.com/superlinked/sie/commit/72a9924fb48894b44711ae9a92c9b919d3039273))
* **mcp:** window GLiNER extract and redact by words so the whole text is read ([#359](https://github.com/superlinked/sie/issues/359)) ([35e711b](https://github.com/superlinked/sie/commit/35e711bf6bde4ff0694b56e58d87b698b93d15ff))
* **server:** bound subword length of long words in GLiNER adapters ([#380](https://github.com/superlinked/sie/issues/380)) ([894a017](https://github.com/superlinked/sie/commit/894a0175749e4da71c6239dc4ea1c6bbaf8ebfae))
* **server:** limit the label prompt of GLiNER-family extract requests ([#381](https://github.com/superlinked/sie/issues/381)) ([5aa0287](https://github.com/superlinked/sie/commit/5aa0287c66e4916f0919660eef6348d0bfbc98b6))
* **server:** reject oversized item text at encode, score, and extract ingress ([#366](https://github.com/superlinked/sie/issues/366)) ([7bd6192](https://github.com/superlinked/sie/commit/7bd61929a0438be8bd785ef161885f54f7169a75))
* **server:** split GLiNER2 documents in linear time ([#371](https://github.com/superlinked/sie/issues/371)) ([359c4f0](https://github.com/superlinked/sie/commit/359c4f08a9005a7e24dd9b43ee32db7d0c43665c))
* **sparse:** score the three cards the page shows and the run behind them ([#346](https://github.com/superlinked/sie/issues/346)) ([c93f18a](https://github.com/superlinked/sie/commit/c93f18a089a4f5cac8b0f1ae80d0c559ac3b7622))


### Performance Improvements

* **gliformer:** measure each task prompt once and skip the per-request eval() ([#365](https://github.com/superlinked/sie/issues/365)) ([d41ba7f](https://github.com/superlinked/sie/commit/d41ba7fb828cedb6a08f4c4028c6ce621385168f))
* **models:** load three GLiClass models with bucketed CUDA graphs ([#372](https://github.com/superlinked/sie/issues/372)) ([a3a6429](https://github.com/superlinked/sie/commit/a3a642981dddd3d6e6c1e2a57ca625e58a62538f))

## [0.8.2](https://github.com/superlinked/sie/compare/v0.8.1...v0.8.2) (2026-09-23)


### Bug Fixes

* **release:** rehearse all artifacts before publication ([#342](https://github.com/superlinked/sie/issues/342)) ([cf2b732](https://github.com/superlinked/sie/commit/cf2b732028703eb6f20cf7db144664f2216eb7d4))

## [0.8.1](https://github.com/superlinked/sie/compare/v0.8.0...v0.8.1) (2026-09-23)


### Features

* **examples:** rebuild the chat example around the recorded multi-turn run ([#338](https://github.com/superlinked/sie/issues/338)) ([25c857e](https://github.com/superlinked/sie/commit/25c857e0ea1a639884ff1f9abbf50d20113327f3))


### Bug Fixes

* **release:** repair audio and CUDA 13 build checks ([#340](https://github.com/superlinked/sie/issues/340)) ([605b3fe](https://github.com/superlinked/sie/commit/605b3fea9e6e8630cf431dae0943fce529540c44))

## [0.8.0](https://github.com/superlinked/sie/compare/v0.7.3...v0.8.0) (2026-09-23)


### ⚠ BREAKING CHANGES

* **server:** `extra_launch_args` may not carry `--nccl-port`, `--host` or `--port`, in any unambiguous abbreviation or `--flag=value` spelling. A profile carrying one fails to load, and the config service refuses the write that would create it, so profiles written straight through the config API need the same edit as the ones shipped in a chart. Replace `--nccl-port` with the `adapter_options.loadtime.nccl_port` option, which applies above `tensor_parallel_size: 1` and is reserved before the flag is passed. Remove `--host` and `--port`: the server passes both for the HTTP listener it talks to.
* **server:** a load-time option that no adapter constructor accepts is refused when the model loads instead of being ignored. `extra_launch_args` may not carry placement flags (including unambiguous abbreviations) and `extra_env` may not set `CUDA_VISIBLE_DEVICES`, `NVIDIA_VISIBLE_DEVICES` or `ROCR_VISIBLE_DEVICES`. The SGLang embedding adapter no longer accepts `pooling_method`. SGLang launch arguments spell the width as `--tensor-parallel-size` for every profile.

### Features

* carry inline video through the queued chat completions path ([#329](https://github.com/superlinked/sie/issues/329)) ([d0633d1](https://github.com/superlinked/sie/commit/d0633d141aa0ce088bed3f447d228c21658e94f5))
* **ci:** add CI and release automation ([4e110b9](https://github.com/superlinked/sie/commit/4e110b9ca86aaf10476ebda3cb789fc9209994b4))
* **config:** add GLM-5.3-Flash across eight accelerators ([#324](https://github.com/superlinked/sie/issues/324)) ([a913d0c](https://github.com/superlinked/sie/commit/a913d0cfd4d5110f685d277d9941c5d081a9ee43))
* **examples:** add detect, image-classify and speech-to-text evaluations ([#313](https://github.com/superlinked/sie/issues/313)) ([8b4f3d4](https://github.com/superlinked/sie/commit/8b4f3d460dec5ceabe1a09872f82996ae8b743fe))
* **examples:** add five recorded task evaluations, with evidence on HuggingFace ([#310](https://github.com/superlinked/sie/issues/310)) ([7d6e11d](https://github.com/superlinked/sie/commit/7d6e11d88452cb9cc72c378914e9625edfdc6032))
* **examples:** add image-search and visual-document-search evaluations ([#314](https://github.com/superlinked/sie/issues/314)) ([c303dfc](https://github.com/superlinked/sie/commit/c303dfc7f7a0c0000bc6338ca20fcf40e6ee2678))
* **examples:** add ocr-two-stage, a recorded two-stage OCR evaluation ([#303](https://github.com/superlinked/sie/issues/303)) ([4202f7d](https://github.com/superlinked/sie/commit/4202f7d96ab84261d8fb88c1a32d0179ea132e34))
* **examples:** add three recorded task evaluations, with evidence on HuggingFace ([#311](https://github.com/superlinked/sie/issues/311)) ([8be1e04](https://github.com/superlinked/sie/commit/8be1e04a78bf2c347d10076b97b1a840b7d23d02))
* **examples:** measure whether a second call earns its place on doc field extraction ([#336](https://github.com/superlinked/sie/issues/336)) ([f4521a9](https://github.com/superlinked/sie/commit/f4521a9077dcd334c138e8372b9088469d39f82b))
* **examples:** re-derive the yes-or-no figure for /structured-output ([#335](https://github.com/superlinked/sie/issues/335)) ([8333afa](https://github.com/superlinked/sie/commit/8333afa28f5bf6bc151b0cf5bbadab06fc1f9713))
* **examples:** score four guardrail models on the same twelve inputs ([#319](https://github.com/superlinked/sie/issues/319)) ([6794e34](https://github.com/superlinked/sie/commit/6794e348ea50d995ad8da7a611b58ec70da92e4e))
* **examples:** score grounded answers in chat, move the incident run to its own example ([#317](https://github.com/superlinked/sie/issues/317)) ([5cc5580](https://github.com/superlinked/sie/commit/5cc5580f110092eeffd67ce5b6bfe8db12311c60))
* **helm:** add a tested high-availability values composition ([#274](https://github.com/superlinked/sie/issues/274)) ([0560f01](https://github.com/superlinked/sie/commit/0560f017155fa4097ec9da0bd456100e4f33aca5))
* **helm:** make the worker /dev/shm size configurable per pool ([#302](https://github.com/superlinked/sie/issues/302)) ([e203682](https://github.com/superlinked/sie/commit/e2036826329065ee35cd84fccba696656a379cee))
* **models:** add google/translategemma-4b-it ([40ae372](https://github.com/superlinked/sie/commit/40ae37238b6a09b0abfeeaa6fbf645d292edcced))
* **models:** add google/translategemma-4b-it ([979961b](https://github.com/superlinked/sie/commit/979961b840604ef04ccc202bbd081e3534fb4552))
* **models:** add Qwen3-VL-8B-Instruct with text, image, and video input ([#323](https://github.com/superlinked/sie/issues/323)) ([92bd4bb](https://github.com/superlinked/sie/commit/92bd4bba8c93a026bd243780daf40e74e273eb13))
* **server:** accept inline video in local chat completions ([#321](https://github.com/superlinked/sie/issues/321)) ([0fbe99b](https://github.com/superlinked/sie/commit/0fbe99b446fe393e79088f9b3c927a40e52cf8b8))
* **server:** move the CUDA 13 SGLang bundle to 0.5.20 for glm5_next ([#320](https://github.com/superlinked/sie/issues/320)) ([ac2358c](https://github.com/superlinked/sie/commit/ac2358c7ed99e7f21943e089da90e3c35fbe7c1f))
* **server:** parse and force GLM tool calls on the queued route ([#328](https://github.com/superlinked/sie/issues/328)) ([84b1a7f](https://github.com/superlinked/sie/commit/84b1a7f8c6eb932af02516deae55d3b87aea63b4))
* **server:** serve one model across several GPUs with tensor parallelism ([#282](https://github.com/superlinked/sie/issues/282)) ([a03f5db](https://github.com/superlinked/sie/commit/a03f5dbf878db47b6a8418c6c4b9f530b486ff88))


### Bug Fixes

* **api:** constrain native stream execution evidence pairs ([7cfa50e](https://github.com/superlinked/sie/commit/7cfa50e73068354808f5b6430c248c494ec2216e))
* **api:** restrict stream evidence to successful terminal chunks ([f256d4e](https://github.com/superlinked/sie/commit/f256d4e496f066f5cb510e18e3f9ad61aa337387))
* bound diagnostic decoding and preserve grammar refusal metadata ([b70cca0](https://github.com/superlinked/sie/commit/b70cca055565590f88b07c7a65914bb468006bb4))
* **ci:** serialize mise bootstrap to protect Node GPG state ([538daff](https://github.com/superlinked/sie/commit/538daff41ca20ee27a69f26bda73dd190bcff35a))
* **ci:** serialize mise bootstrap to protect Node GPG state ([996b07f](https://github.com/superlinked/sie/commit/996b07feedc2a55cebd6962895e5d7a0d7afc0b8))
* **ci:** serialize mise bootstrap to protect Node GPG state ([#287](https://github.com/superlinked/sie/issues/287)) ([538daff](https://github.com/superlinked/sie/commit/538daff41ca20ee27a69f26bda73dd190bcff35a))
* **config:** add Qwen3.8 grammar fallback ([#258](https://github.com/superlinked/sie/issues/258)) ([1c7bbe8](https://github.com/superlinked/sie/commit/1c7bbe806c1f1fba9929f96e4b4a96b8c462d2fb))
* **config:** keep GLM-5.3-Flash reasoning private on every route ([#326](https://github.com/superlinked/sie/issues/326)) ([1ffb19c](https://github.com/superlinked/sie/commit/1ffb19c4a8b1b42c30dd344a6c85abe036186597))
* document observed images in server generation usage ([81aecec](https://github.com/superlinked/sie/commit/81aececaca31a1518102f209eeaefa878b81c9c2))
* **examples:** record Granite Guardian through chat completions on guardrails ([#316](https://github.com/superlinked/sie/issues/316)) ([86f4e1e](https://github.com/superlinked/sie/commit/86f4e1ed56b317614e040fd88548341a852a28c9))
* **gateway:** keep the served surface when an authoritative export shrinks it ([#270](https://github.com/superlinked/sie/issues/270)) ([c65aee5](https://github.com/superlinked/sie/commit/c65aee57224d93c036490f601092bf2c8e165ee9))
* **gateway:** pass the invalid_guard_verdict worker code through ([#297](https://github.com/superlinked/sie/issues/297)) ([7a8b09b](https://github.com/superlinked/sie/commit/7a8b09b6db1c206e21bb2b85b7af415e5f644802))
* **gateway:** refuse requests when the auth configuration is an error ([#269](https://github.com/superlinked/sie/issues/269)) ([8c84419](https://github.com/superlinked/sie/commit/8c84419ac4c4882b081aeda3413cd66b86dab191))
* **generate:** preserve image usage and complete execution evidence ([d266205](https://github.com/superlinked/sie/commit/d2662052bd53a017b300a64a246f4989f4f8fe30))
* **generate:** preserve image usage and complete execution evidence ([c9e9b7e](https://github.com/superlinked/sie/commit/c9e9b7e117ab8de4128a6ab10e7ce2be20734352))
* **generate:** preserve image usage and complete execution evidence ([#286](https://github.com/superlinked/sie/issues/286)) ([d266205](https://github.com/superlinked/sie/commit/d2662052bd53a017b300a64a246f4989f4f8fe30))
* **generation:** require explicit streamed candidate indexes ([c0aca44](https://github.com/superlinked/sie/commit/c0aca4438f1fe1359a8a0ef704ef592e49dd193f))
* **helm:** make KEDA hook Job resources configurable ([#268](https://github.com/superlinked/sie/issues/268)) ([5e726f6](https://github.com/superlinked/sie/commit/5e726f64be7150f47fb08397d4e444ed0450030c))
* identify unsupported Outlines JSON Schema type values ([729cadc](https://github.com/superlinked/sie/commit/729cadcfb5bd7f75c7af28899efe93ca319d794c))
* identify unsupported Outlines JSON Schema type values ([0c0408b](https://github.com/superlinked/sie/commit/0c0408b1de443f7309c71660bdd3cccd29ca37d0))
* identify unsupported Outlines JSON Schema type values ([#289](https://github.com/superlinked/sie/issues/289)) ([729cadc](https://github.com/superlinked/sie/commit/729cadcfb5bd7f75c7af28899efe93ca319d794c))
* isolate grammar followers and type malformed backend errors ([4c5e05f](https://github.com/superlinked/sie/commit/4c5e05fe1ce5cd4fafe99f4e79df6c83b6b7a0b8))
* **mcp:** parse the Host header before trusting it for the OAuth origin ([27ae3a1](https://github.com/superlinked/sie/commit/27ae3a18033a09a1c9c3f18de6bcef1d51f37db4))
* **mcp:** parse the Host header before trusting it for the OAuth origin ([604449a](https://github.com/superlinked/sie/commit/604449a4a435c0bc4fe2bb2957a1758f7d1e27f8))
* **mcp:** stop deriving the OAuth origin from X-Forwarded headers ([#292](https://github.com/superlinked/sie/issues/292)) ([f5c4451](https://github.com/superlinked/sie/commit/f5c4451f1cfc9a2d98dffc5cb94b08ad2d38efa0))
* **mcp:** validate bracketed IPv6 before advertising origins ([a66f6a0](https://github.com/superlinked/sie/commit/a66f6a096153e7dd8c41197e74bc8d9238c1823c))
* **mcp:** validate Host before parsing OAuth request origins ([d9507f5](https://github.com/superlinked/sie/commit/d9507f5a896ccdc2d0c8ffaa89e7b39d5c95b523))
* **models:** bound translategemma prompts to the documented 2K input context ([ac810ad](https://github.com/superlinked/sie/commit/ac810ad01fb2e9744b47a29c7537d28f4e4e520a))
* **models:** pin MADLAD to greedy sampling by default ([b0f0eeb](https://github.com/superlinked/sie/commit/b0f0eebf4370d6f151eba98cc1704750617996a5))
* **models:** pin MADLAD to greedy sampling by default ([a968874](https://github.com/superlinked/sie/commit/a968874ce724c4b9dc5c8a61313188655705f4e9))
* **models:** pin the triton attention backend for translategemma ([2dfe7b2](https://github.com/superlinked/sie/commit/2dfe7b25778f67ac37083a2e1244f627fce3e6c4))
* omit logprobs for rewritten guard verdicts ([ba8c3b1](https://github.com/superlinked/sie/commit/ba8c3b11e98711b974bef5f07d8ba15a6e5bcbbd))
* preserve generation progress and reject invalid guard verdicts ([424f7ae](https://github.com/superlinked/sie/commit/424f7ae15dc1fe4e7e66a171160322f38264dedd))
* preserve generation progress and reject invalid guard verdicts ([398c673](https://github.com/superlinked/sie/commit/398c673885bd94e0c5f047240ae3289a04f72754))
* preserve generation progress and reject invalid guard verdicts ([#285](https://github.com/superlinked/sie/issues/285)) ([424f7ae](https://github.com/superlinked/sie/commit/424f7ae15dc1fe4e7e66a171160322f38264dedd))
* preserve grammar-safe generation defaults ([#261](https://github.com/superlinked/sie/issues/261)) ([3b16ffb](https://github.com/superlinked/sie/commit/3b16ffbca4ee5fa816f9636b9b998a3e740830ff))
* preserve TensorRT-LLM completion text and stop alignment ([#267](https://github.com/superlinked/sie/issues/267)) ([d345bc0](https://github.com/superlinked/sie/commit/d345bc0a0663e587687ec535c75d3aa7004f844b))
* **release:** format generated SDK package metadata ([#339](https://github.com/superlinked/sie/issues/339)) ([36bd49e](https://github.com/superlinked/sie/commit/36bd49e15f6817bbb26d91f988fcba8cab3a49c4))
* **sdk:** declare terminal streaming execution evidence ([0261183](https://github.com/superlinked/sie/commit/026118304fdd167a224678ac93bf0e7f70a124e7))
* **sdk:** declare terminal streaming execution evidence ([#283](https://github.com/superlinked/sie/issues/283)) ([0261183](https://github.com/superlinked/sie/commit/026118304fdd167a224678ac93bf0e7f70a124e7))
* **sdk:** document terminal streaming execution evidence ([f3a05ba](https://github.com/superlinked/sie/commit/f3a05ba287f4d5a96afe4d5f72be3abceacdd352))
* **server:** bound native encode, score and extract request bodies ([#271](https://github.com/superlinked/sie/issues/271)) ([8479f1c](https://github.com/superlinked/sie/commit/8479f1c8beff090da711456dcddbe03abb899965))
* **server:** evict the least recently used group when two blocks cost the same ([#294](https://github.com/superlinked/sie/issues/294)) ([120059f](https://github.com/superlinked/sie/commit/120059fd632d119c6d8cb91bcf96068ef452515e))
* **server:** fetch model weights outside the registry load lock ([#272](https://github.com/superlinked/sie/issues/272)) ([394df63](https://github.com/superlinked/sie/commit/394df634824eb7cde4fafa38eee70cc46258eed7))
* **server:** force complete GLM argument pairs and reject oversized GLM calls ([#330](https://github.com/superlinked/sie/issues/330)) ([079647d](https://github.com/superlinked/sie/commit/079647d52261063aa7fa9066b1393a14d0b2976c))
* **server:** keep inline media payloads out of SGLang child logs ([#325](https://github.com/superlinked/sie/issues/325)) ([d8573f2](https://github.com/superlinked/sie/commit/d8573f24fc357b3e57df94c06ac7a12f2c6784b0))
* **server:** map GroundingDINO detections back onto the caller's labels ([#277](https://github.com/superlinked/sie/issues/277)) ([958414c](https://github.com/superlinked/sie/commit/958414c5b376bb370da608b6bbb957b20556457e))
* **server:** narrow validated terminal result type ([56efe5f](https://github.com/superlinked/sie/commit/56efe5fbe2abf18f30f3d19d9feb14cbe04b6ee4))
* **server:** pin cuda-tile for TensorRT-LLM ([#254](https://github.com/superlinked/sie/issues/254)) ([6484fab](https://github.com/superlinked/sie/commit/6484fabdcf5365ff9e4d0381e95a301a524e7ab4))
* **server:** read Qwen2.5-style tool calls as Hermes JSON on the queued route ([#332](https://github.com/superlinked/sie/issues/332)) ([d8494f3](https://github.com/superlinked/sie/commit/d8494f3ff9642ebdd0ccf06c1d8c03fc23badb97))
* **server:** refuse listener flags in extra_launch_args ([#304](https://github.com/superlinked/sie/issues/304)) ([7a68262](https://github.com/superlinked/sie/commit/7a68262a0f05e6cc42790b77fc779e58ec2696e0))
* **server:** reject failed SGLang generation terminals ([829f80c](https://github.com/superlinked/sie/commit/829f80ca4a5dceea1defc9c9891b6e1111bdf9e7))
* **server:** reject failed SGLang generation terminals ([3bc140d](https://github.com/superlinked/sie/commit/3bc140d29214c8410267771f24b988da59864ebf))
* **server:** reject failed SGLang generation terminals ([#284](https://github.com/superlinked/sie/issues/284)) ([829f80c](https://github.com/superlinked/sie/commit/829f80ca4a5dceea1defc9c9891b6e1111bdf9e7))
* **server:** reject malformed backend finish metadata ([2be0155](https://github.com/superlinked/sie/commit/2be0155f8ef1ff0eda201739e87fca1a0ed5d6e4))
* **server:** report a valueless --mm-process-config instead of raising ([#333](https://github.com/superlinked/sie/issues/333)) ([2f7ad9d](https://github.com/superlinked/sie/commit/2f7ad9db0dff6781210bb28bf3567eff2436a9f1))
* **server:** require a usable startup budget for SGLang profiles ([#301](https://github.com/superlinked/sie/issues/301)) ([038a8d9](https://github.com/superlinked/sie/commit/038a8d9116a0bf6f776809c9d8355b3741f5d4dc))
* **server:** require complete generation candidates ([afaab57](https://github.com/superlinked/sie/commit/afaab574e6d035f399fce98029281e9c2f2ef507))
* **server:** ship FFmpeg shared libraries in the SGLang runtime image ([#322](https://github.com/superlinked/sie/issues/322)) ([9d98037](https://github.com/superlinked/sie/commit/9d9803790a7c582da0d7232be326177a78508d92))
* **server:** tell SGLang when a native generate request starts inside reasoning ([#327](https://github.com/superlinked/sie/issues/327)) ([26702c7](https://github.com/superlinked/sie/commit/26702c7a9e5b78326692b2c811c36f91ad8c9db1))
* **server:** validate video pixel and frame counts before the budget arithmetic ([#337](https://github.com/superlinked/sie/issues/337)) ([703841a](https://github.com/superlinked/sie/commit/703841a3871cbd0838b6c3be93886e8cc603241a))
* **tasks:** stop expanding possibly-empty arrays bare under set -u ([#278](https://github.com/superlinked/sie/issues/278)) ([e0084c7](https://github.com/superlinked/sie/commit/e0084c78c3cd67b044c4968fc2d2212967a74bb3))
* **toolchain:** upgrade Rust to 1.98.1 and patch rustls ([#276](https://github.com/superlinked/sie/issues/276)) ([a1f6fab](https://github.com/superlinked/sie/commit/a1f6fab356cbff88609a7d999c8e3b3fd02e2365))
* **tooling:** use canonical Rust LLVM component ([#256](https://github.com/superlinked/sie/issues/256)) ([2183bc1](https://github.com/superlinked/sie/commit/2183bc1c3e7b034a8219e8446f99954f601eb87e))
* **ts-sdk:** decode base64 data URLs with media-type parameters ([#247](https://github.com/superlinked/sie/issues/247)) ([21289a1](https://github.com/superlinked/sie/commit/21289a1798a54fcce1b556bcd01ec0b07ff025c2))
* validate complete first guard verdict evidence ([514b07f](https://github.com/superlinked/sie/commit/514b07fffc4b5f84985d5861cb12fc1de0564fd5))
* validate unsupported template placeholders ([#249](https://github.com/superlinked/sie/issues/249)) ([0508e17](https://github.com/superlinked/sie/commit/0508e17aa323c7773be0d9d58c85038d7d9b043e))

## v0.7.3 (2026-09-03)

### Highlights

This release adds MADLAD translation and improves how generation requests finish, fail, and cancel.

### Features

- Translate text with `google/madlad400-3b-mt`, served through CTranslate2 with native request batching. The initial profile uses FP32 and supports up to 512 input and 512 output tokens.
- An opt-in TensorRT-LLM bundle provides an encoder-decoder adapter on CUDA 13. It is not the default backend for the model catalog.
- The Rust/Candle worker implementation is now included, covering native embeddings and ColBERT scoring.
- Converted serving artifacts can be pinned and verified before loading, then reused from the local cache for offline serving.

### Bug fixes

- Cancelled generation streams now close their upstream requests, and model unloading drains active generation before releasing the runtime.
- Model-load failures and generation errors remain errors through streaming and buffered responses. Python and TypeScript clients retain useful error codes, invalid-parameter details, and validated retry hints without exposing raw backend diagnostics.
- Streaming support is reported in model capabilities. Unsupported streaming requests are rejected before execution.
- Encoder-decoder models enforce separate input and output limits instead of incorrectly treating them as one shared context window.

## v0.7.2 (2026-08-27)

### Highlights

Qwen3.8 joins the model catalog, Alibaba Cloud deployments gain native object-storage support, and a broad set of fixes improves SDK errors, batching, and self-hosted reliability.

### Features

- Added `Qwen/Qwen3.8-27B-FP8` for text and image generation, with tool calling and structured output. Alongside the conservative default profile, explicit H100, H200, and RTX PRO 6000 profiles support a 256K context window. These profiles return answers without a separate thinking mode.
- Added native Alibaba Object Storage Service support for queued payloads and SDK storage access, plus ACK Helm settings and RRSA workload identity.
- Self-hosted clusters can configure file-backed and replicated NATS work queues instead of relying only on the default in-memory, single-replica queue.

### Bug fixes

- Python and TypeScript SDKs detect incomplete batch responses instead of silently pairing results with the wrong inputs. Mixed text-and-image batches retain their input ordering, and multimodal embedding models batch text-only inputs correctly.
- Generation failures, including empty output and model-loading errors, are surfaced consistently. Retry handling better distinguishes a model still loading from a permanent failure, and stream errors retain their request identifiers.
- The OpenAI-compatible embeddings endpoint rejects unsupported `dimensions` values instead of silently returning a different vector width. Cold loads return a retryable response rather than holding the request open throughout model loading.
- Invalid image data receives a useful input error, unknown model identifiers include nearby matches, and non-finite reranker scores are rejected.
- Model eviction no longer leaves queued requests waiting indefinitely or blocks unrelated requests while a worker shuts down. Failed engine starts release their ports and report the underlying crash instead of a misleading timeout.
- Helm deployments gain gateway disruption protection and shutdown budgets, ingress timeouts aligned with model loading, and fixes for KEDA scale-to-zero checks and ACK storage permissions.
- ColPali and ColQwen checkpoint revisions are pinned correctly for repeatable loading.

### Performance improvements

- SGLang compiler caches can persist across worker restarts, with separate cache entries for incompatible GPU and runtime combinations.
- Qwen3.8 hardware profiles tune speculative decoding and scheduling.

### Upgrade notes

- The CUDA 13 `gemma` bundle is renamed to `sglang-cu130`. Update explicit bundle selections and image references from `cuda13-gemma` to `cuda13-sglang-cu130`.
- The TypeScript SDK adds the explicit `timeoutMs` option. The older `timeout` option remains a millisecond-based alias.
- File-backed queues are opt-in. Changing the setting does not convert existing NATS streams; existing streams need a planned drain and recreation. Replication also requires a NATS cluster.

## v0.7.1 (2026-08-09)

Docling OCR now explicitly uses the English recognizer, fixing English words being joined together. GLiNER rejects blank documents and label prompts that leave no room for document text before running extraction.

## v0.7.0 (2026-08-08)

### Bug fixes

- Python and TypeScript clients handle malformed generation responses and unexpected redirects as clear errors, without copying response bodies into diagnostics.
- Queued extraction results retain the input item identifier, so callers can associate results with their documents.
- Grounding DINO normalizes free-form detection prompts to the format expected by the model.
- Models selected through `SIE_PINNED_MODELS` remain resident when starting the server. Inconsistent pinned-model and model-filter settings are rejected at startup.

## v0.6.30 (2026-08-07)

Qwen3.6 thinking profiles enable CUDA graphs and overlapping scheduling while retaining non-speculative decoding for reasoning and structured-output compatibility.

## v0.6.29 (2026-08-07)

Gemma 4 thinking profiles gain speculative decoding with the matching assistant model. Separate non-speculative profiles remain available for structured output, preserving the selected context window and thinking mode.

## v0.6.28 (2026-08-07)

### Features

- Qwen3.6 and Gemma 4 non-thinking profiles gain tuned speculative-decoding configurations for long-context generation.
- Structured-output routing can select a profile-specific, non-speculative counterpart that preserves the requested context window and thinking mode.

### Bug fixes

- Speculative draft models use pinned revisions and are prepared alongside locally cached main models. An unrelated cached revision no longer satisfies an explicitly requested model revision.

## v0.6.27 (2026-08-06)

### Highlights

Generation gained a direct-server Responses endpoint and more reliable handling of model-specific reasoning, while the retrieval catalog expanded with multilingual dense and late-interaction models.

### Features

- Added `lightonai/mLateOn` for multivector embeddings and scoring, and `ibm-granite/granite-embedding-97m-multilingual-r2` for dense embeddings.
- Added stateless, non-streaming text requests at the direct server's `/v1/responses` endpoint, with explicit errors for unsupported options.
- Added hardware-specific long-context and thinking profiles for selected Qwen and Gemma models. Context and output limits remained profile-specific; the default model settings were not expanded globally.
- Enabled generation streams over the local worker-ingest connection, including request cancellation.

### Bug fixes

- Prevented Qwen and Gemma reasoning blocks from appearing in visible answers when thinking is disabled, including delimiters split across streamed chunks.
- Enabled Python clients to retry capacity failures received before generation starts, including failures delivered inside an SSE stream.
- Kept Docling's normal document parsing available when optional OCR assets fail to initialize.
- Corrected native extraction image preprocessing and preserved Qwen3-VL reranker batching with newer Transformers versions.
- Redirected the server's outdated root playground to its interactive API documentation.

## v0.6.26 (2026-08-02)

Added H100 FP8 generation profiles for `Qwen/Qwen3.6-27B`, `Qwen/Qwen3.6-35B-A3B`, and `google/gemma-4-31B-it`. The newly added 35B Qwen and 31B Gemma configurations initially exposed an 8K context window, rather than their checkpoints' full native context.

## v0.6.25 (2026-07-30)

### Features

- Added `Qwen/Qwen3-Embedding-8B` for text embeddings.

### Bug fixes

- Added early validation of OpenAI-compatible embedding requests containing more than 256 inputs to prevent oversized result payloads.
- Applied supported sequence-length limits to GLiNER and NuNER model configurations.
- Bounded recovery work when an invalid input affects a shared encoding batch, preventing repeated decoding failures from triggering unbounded retries of smaller batches.

## v0.6.24 (2026-07-26)

### Features

- Added exact adapter-revision pinning to LoRA entries in model configuration for reproducible loading.

### Bug fixes

- Included missing tokenizer dependencies in the affected model configurations.

### Performance improvements

- Cached MUVERA projection state across requests, avoiding reconstruction of the same random projection structures for every encoding operation.

## v0.6.23 (2026-07-24)

### Highlights

The direct server's `/v1/generate` endpoint gained image-input and structured-output support. Python and TypeScript SDKs also gained helpers for image inputs, grammars, and Responses; the Responses helpers covered stateless, non-streaming text requests.

### Features

- Added `naver/v-splade-quality` for sparse text and image embeddings.

### Bug fixes

- Corrected context-sensitive structured-output schema validation and made malformed grammar errors consistent.
- Restored trained ColBERT projections and query-expansion behavior, including the ColBERTv2 and Jina ColBERT retrieval recipes.
- Serialized concurrent tokenizer and hidden-state operations that could otherwise interfere with one another.
- Preserved pinned revisions when loading trusted model code, including fallback paths.
- Removed unsupported visual MUVERA profiles from the advertised catalog.

## v0.6.22 (2026-07-22)

Docling gained support for verified, immutable model artifacts. Staged file-inventory and hash checks made missing or mismatched assets explicit instead of silently using a different artifact set.

## v0.6.21 (2026-07-21)

### Highlights

This release added a text-reranking compatibility API and expanded the embedding and transcription catalog, alongside fixes to retrieval and vision-model execution.

### Features

- Added Cohere-compatible text-only reranking at `/v1/rerank` and `/v2/rerank`, with strict validation of supported request fields and complete-result handling.
- Added `Snowflake/snowflake-arctic-embed-s` for text embeddings and `openai/whisper-large-v3-turbo` for audio transcription.

### Bug fixes

- Corrected `Alibaba-NLP/gte-Qwen2-7B-instruct` serving to use its checkpoint's bidirectional embedding implementation rather than a causal generation implementation.
- Restored the 512-token capacity of `prithivida/Splade_PP_en_v2` while keeping its retrieval-specific query and document limits separate.
- Prevented concurrent ColPali forwards from interfering with Transformers' output recording and stopped temporary tensors from accumulating between requests.
- Corrected model discovery for an explicitly empty selection, which previously advertised every model in the bundle.
- Added validation for blank reranking inputs and filtered relation results whose endpoints were not selected by the extraction request.

## v0.6.20 (2026-07-18)

### Highlights

The model catalog expanded with compact multilingual retrieval and text-classification options. Generative OCR models gained a shared SGLang serving path with continuous batching.

### Features

- Added `intfloat/multilingual-e5-small`, `tencent/R3-embedding-0.6b`, and `tencent/R3-rerank-0.6b`.
- Added `fastino/gliguard-LLMGuardrails-300M` for text classification.
- Enabled LightOnOCR, PaddleOCR-VL, and GLM-OCR serving through the generative OCR adapter.

### Bug fixes

- Corrected classification-threshold and multi-label handling in GLiNER2-based classification.
- Corrected GLiREL entity offsets and relation text when mapping tokenized inputs back to the original text.

## v0.6.19 (2026-07-14)

### Bug fixes

- Corrected unknown-model responses to return `404` instead of `500`.
- Marked permanently failed model loads as terminal failures, preventing queued requests from waiting for a model that cannot become ready.
- Preserved one multivector result per input, in input order, when encoding with the ColBERT adapter.
- Restored Florence-2 processor compatibility and pinned offline chat-template rendering to the configured tokenizer revision.
- Protected cross-encoder tokenization from concurrent access during inference and token counting.

## v0.6.18 (2026-07-12)

Gateway configuration diagnostics were updated to redact API bearer tokens, administrator tokens, and configuration-service credentials instead of including their values in debug output.

## v0.6.17 (2026-07-09)

### Features

- Added support for multiple GPU worker children behind one sidecar, distributing work according to each child's readiness and queue pressure.
- Added `vidore/colSmol-256M` for text and image multivector embeddings, with an optional MUVERA representation.

### Bug fixes

- Kept worker IPC health checks responsive when GPU health inspection fails.
- Released cancelled scheduler reservations and bounded queue admission through completion, preventing cancelled work from leaving a worker appearing permanently busy.

## v0.6.16 (2026-07-07)

### Bug fixes

- Restored NV-Embed-v2's native encoding recipe, including its instruction-aware latent-attention pooling, instead of generic embedding pooling.
- Applied trained PyLate Dense projection chains in ColBERT adapters and aligned the ModernBERT implementation with the checkpoint's forward pass.
- Restored query and document prefixes on the SentenceTransformer profile for `intfloat/multilingual-e5-large`.
- Fixed scale-from-zero handling for GPU-agnostic requests in multi-profile pools and normalized explicit GPU demand labels consistently.

## v0.6.15 (2026-07-03)

### Highlights

Apple Silicon gained an MLX generation backend for configured models, including Qwen3.5-4B. Self-hosted Helm deployments gained opt-in distributed tracing with a bundled OpenTelemetry collector and optional Tempo backend.

### Bug fixes

- Restored the checkpoint-specific embedding recipes for EmbeddingGemma, GTE-Qwen2, and Stella models, including their tokenizer and query-instruction handling.
- Applied profile runtime options consistently on queued encoding requests and preserved the distinction between unsigned-byte and binary embeddings.
- Corrected out-of-memory errors on the OpenAI-compatible embeddings endpoint to return `503 RESOURCE_EXHAUSTED`.
- Bounded continuous-batch draining so a busy LoRA adapter cannot indefinitely delay other adapters.
- Improved cleanup during model unload and serialized hot-reload changes to the model registry.
- Preserved trace context through embedding rewrites and queue dispatch.
- Prevented late replies from abandoned generation attempts from replacing the active result.

## v0.6.14 (2026-06-26)

Fixed ColBERT query–document pair scoring, including the ModernBERT and rotary variants, so retrieval results can be reranked without server errors. MUVERA encoding now returns the requested dense vectors instead of dropping them from the response.

## v0.6.13 (2026-06-25)

Live model-configuration updates now preserve profile-qualified variants, validate changes before applying them, and unload removed variants safely. Cleanup of large request payloads now tracks the exact stored objects and retries failed deletions.

MCP tools now support document summarization, entity extraction, and masking detected PII. Large-text extraction uses bounded overlapping chunks, and operators can choose models and GPU routing separately for different tools.

## v0.6.12 (2026-06-24)

GLiClass now returns an actionable input-too-long error when the combined label and text input exceeds its context window. The gateway also avoids unnecessary storage deletions for requests that were never offloaded to object storage.

## v0.6.11 (2026-06-23)

### Highlights

This release expands text generation with Gemma 4 and makes self-hosted capacity easier to keep ready for requests.

### Features

- Added Gemma 4 E2B, E4B, and 26B-A4B model configurations in a dedicated `gemma` bundle using CUDA 13.
- Pinned models are now loaded on assigned workers and protected from idle and memory-pressure eviction.
- Logical resource pools can use existing worker capacity through a configurable backing queue pool, with Python and TypeScript SDK support.

### Bug fixes

- Generation requests made while a model is loading receive a retryable response. Queue retries preserve their delivery limits, and worker fallback routing respects the requested pool.
- Helm reports incompatible bundle/platform selections before deploying workers with an unavailable image.

## v0.6.10 (2026-06-22)

### Highlights

This release corrects embedding and object-detection outputs and makes pending generation work visible in the model and cluster-status APIs.

### Features

- Added a self-hostable MCP server for document conversion, image descriptions, document question answering, and structured output. It uses SIE's public inference APIs and requires a configured SIE endpoint, the appropriate models, and authentication settings. Question answering works on the documents supplied to each call, without a persistent index.

### Bug fixes

- Qwen3-VL embeddings now use the model's final normalized hidden state and consistent instruction formatting, including a default for blank instructions.
- Florence-2 detection results now return pixel-space `[x, y, width, height]` bounding boxes.
- Python installations can resolve the `transformers5` bundle without conflicting with the server's dependency constraints.

### Breaking changes

- The bundled Florence-2 base-ft and large configurations now default to object detection instead of OCR. Set the extraction task explicitly if you need OCR.
- Removed the bundled `naver-clova-ix/donut-base-finetuned-rvlcdip` model configuration.
- GLiNER v2.5 default entity thresholds changed to `0.60` for small, `0.55` for medium, and `0.75` for large. Set an explicit threshold to preserve previous extraction behavior.

## v0.6.9 (2026-06-19)

Added per-pool pinned-model settings to the pool API and Python SDK, including profile-qualified model IDs. Helm model preloading now respects each worker's pool, hardware profile, and bundle. LoRA adapters also receive compatible adapter names when their model IDs contain characters that PEFT cannot use directly.

## v0.6.8 (2026-06-16)

Self-hosted pools can now enforce a minimum number of warm workers through KEDA. Active rerankers are no longer mistaken for idle models, model unloading is coordinated with in-flight work, and multimodal scoring accounts for media when sizing batches.

## v0.6.7 (2026-06-16)

### Features

- Added a 32K-token serving profile for Qwen3.6-27B on RTX PRO 6000. The base model configuration retains its 4K context window.

### Bug fixes

- Grammar-constrained Qwen3.5-4B requests use a non-speculative profile so structured-output constraints are enforced.
- Long model startups now use consistent readiness timeouts across the gateway, workers, and Helm configuration. Reapplying unchanged model configuration no longer needlessly unloads models.

### Performance improvements

- LightOnOCR processes multiple pages in bounded batches, preserving page order and handling different image sizes.

## v0.6.6 (2026-06-14)

Fixed false configuration mismatches between the configuration service and workers when models inherit profiles or belong to specific pools and bundles. Configuration checks also recover when previously missing bundle metadata becomes available.

## v0.6.5 (2026-06-13)

### Bug fixes

- Image-based reranking now transports Python SDK image inputs correctly and includes document images in the Qwen3-VL reranker's prompt.
- Structured generation correctly resolves JSON Schema references without discarding constraints beside a reference.
- SGLang model loading no longer blocks the server event loop, allowing health checks and other requests to remain responsive during startup.
- Configuration updates correctly account for pool ownership, replace stale snapshots, and stop advertising removed model configurations as ready.

### Breaking changes

- Provisioning responses now use HTTP `503` with retry information instead of HTTP `202`. The Python and TypeScript SDKs understand the updated response; direct API clients should handle the new status when waiting for capacity.

## v0.6.4 (2026-06-11)

Added an AKS Helm overlay with Azure Workload Identity support. Self-hosted clusters also recover worker health subscriptions after a stale NATS connection, and binary request data is preserved when passed from the sidecar to the inference worker.

## v0.6.3 (2026-06-10)

### Highlights

Chat requests can now combine text and images, while the model catalog gains more classification, extraction, and object-detection options.

### Features

- The gateway's OpenAI-compatible chat endpoint now accepts inline image data for vision-capable models while preserving text/image ordering.
- Added ModernBERT-base-zeroshot-v2.0 and BART-large-MNLI classification configurations, GLiNER2-large-v1 extraction, and OWLv2-large-patch14-ensemble object detection.
- Added Azure Blob support for model caching and large request payloads.

### Bug fixes

- Corrected image preprocessing and removed padding from document embeddings for `nvidia/llama-nemoretriever-colembed-3b-v1`.
- Workers are removed promptly from gateway discovery when they shut down, reducing routing to stale workers.

## v0.6.2 (2026-06-08)

Added three dense text-embedding models: `mixedbread-ai/mxbai-embed-large-v1`, `Snowflake/snowflake-arctic-embed-l-v2.0`, and `nomic-ai/modernbert-embed-base`.

Self-hosted startup is more reliable: the configuration service can serve health checks while NATS connects, single-profile bundles can scale from zero for requests without an explicit GPU selection, and CUDA images include the build tool needed for SGLang's first-use kernels. Qwen3-VL embedding models also accept and validate their configured output dimension.

## v0.6.1 (2026-06-07)

Added static, non-expiring queue pools in Helm, with startup validation for invalid pool settings. The default worker queue is again shared as `default`, so SDK requests that specify only a hardware profile route correctly; dedicated pools remain explicitly configurable.

## v0.6.0 (2026-06-07)

Queue routing and autoscaling now distinguish each pool, hardware profile, and model bundle, keeping work assigned to the intended worker group.

### Breaking changes

- Queue subjects now use `sie.work.{pool}.{machine_profile}.{bundle}.{model}`. Upgrade the gateway, sidecar, and Helm chart together; the previous subject format is not supported.
- Helm workers now default to their worker-group name as the queue pool. Set `workers.common.queuePool: "default"` explicitly to retain a shared queue.

## v0.5.0 (2026-06-04)

### Highlights

Self-hosted worker pools can now serve multiple bundles with separate replica limits, making it possible to scale embedding and generation workloads independently on the same machine profile.

### Features

- Added Granite Guardian 3.0 2B for content-safety verdicts, with a configurable verdict threshold. Added SQLCoder-7B-2 for completion-based SQL generation using its native prompt format.
- Added configurable `code`, `sql`, and `guard` model aliases and exposed matching capabilities in model metadata. The default `sql` alias uses Qwen3-4B-Instruct-2507, not SQLCoder.
- Added a separate gateway metrics listener for Prometheus scraping without opening inference endpoints to unauthenticated access.

### Bug fixes

- Florence-2 extraction now honors the supplied instruction.
- Guard-model verdict handling now keeps returned log probabilities consistent and rejects unsupported multi-candidate sampling.
- Helm rejects missing or invalid per-bundle replica limits during rendering.

### Breaking changes

- Move each pool's `bundle`, `minReplicas`, `maxReplicas`, `extraEnv`, and `imageBundle` settings into `workers.pools.<pool>.bundles.<bundle>`. The bundle name becomes the map key; `workers.common.bundle` is removed.
- Worker resource names change from `worker-<pool>` to `worker-<pool>-<bundle>`. Upgrades must explicitly remove obsolete StatefulSets, ScaledObjects, PodDisruptionBudgets, and image-prepull DaemonSets so old resources do not interfere with scaling or node drains.

## v0.4.2 (2026-06-03)

### Highlights

This release introduces the Rust worker sidecar for queued inference and expands document extraction and image-text model support.

### Features

- Added MinerU2.5-Pro-2604-1.2B for document OCR and Marqo fashionSigLIP for image-text embeddings. Docling now also accepts image inputs.
- Chat completions accept `min_tokens` and `chat_template_kwargs`; model profiles can supply default sampling settings.
- Added an FP8 serving profile for Qwen3.6-27B on RTX PRO 6000 and increased Qwen3-0.6B's configured context window to 4,096 tokens.
- Workers reconcile configuration changes after missed updates or reconnects.

### Bug fixes

- Kept generation dispatch separate from embedding, scoring, and extraction queues.
- Fixed `dense_dim` handling in CLIP and PyTorch embedding adapters and CUDA-cache cleanup in visual-document adapters.

### Breaking changes

- Cluster inference is now queue-only. Queue workers require the Rust worker sidecar and NATS JetStream; the Helm chart enables the sidecar by default. Custom deployments must include the sidecar alongside the Python inference worker.

## v0.4.1 (2026-05-28)

Added Qwen3.6-27B model support and updated the Linux GPU dependency stack to CUDA 12.9. Generation requests are now dispatched separately from shared inference queues.

## v0.4.0 (2026-05-27)

### Highlights

SIE now serves text generation alongside embeddings, reranking, and extraction, including streaming responses and structured output.

### Features

- Added Qwen3-0.6B, Qwen3-4B-Instruct-2507, and Qwen3.5-4B generation models through SGLang.
- Added a native generation API, OpenAI-compatible chat and legacy completions endpoints, and an initial gateway Responses API implementation. The Python and TypeScript SDKs expose generation options and streaming.
- Added multi-turn tool calls, multiple response candidates, log probabilities, seeded sampling, and per-request LoRA selection. JSON-schema, regex, and grammar-constrained output are available where supported by the selected model and backend.
- Added a browsable API reference at `/docs` and optional bundled certificate management with self-signed TLS for self-hosted clusters.

### Bug fixes

- Streaming now surfaces backpressure failures instead of silently dropping output chunks, and cancellation prevents duplicate generation attempts.
- Fixed decoding of base64 image inputs and included the system libraries needed by Docling in worker images.
- GPU-aware health checks detect unusable CUDA contexts so unhealthy workers can be taken out of service.

## v0.3.4 (2026-05-14)

### Features

- Python and TypeScript clients now expose `InputTooLongError` for extraction inputs that exceed model limits.
- Helm can use the model-cache bucket's `payloads` prefix for large inference payloads.

### Bug fixes

- Fixed gateway startup during concurrent configuration changes and made shared queue routing the default for workers.
- Relaxed the Python SDK's installation requirement to Python 3.12 or later, and removed unnecessary X11 dependencies from image-processing installations.

## v0.3.3 (2026-05-13)

### Highlights

Added ColQwen3 and Nemotron ColEmbed v2 for visual-document retrieval, with clearer failures for oversized extraction inputs and stalled model loading.

### Bug fixes

- GLiClass now enforces the configured overflow policy and returns `INPUT_TOO_LONG` with HTTP 400 instead of crashing on oversized inputs.
- Model loading now distinguishes stalled downloads from time spent loading downloaded weights and applies separate timeout bounds.
- Worker images now include the spatial-index library required by document extraction dependencies.

## v0.3.2 (2026-05-08)

### Features

- Enabled pairwise scoring for the supported ColBERT model variants.
- Added a Docling OCR profile, gateway OpenAPI discovery, configurable Kubernetes probe timing, cert-manager TLS support, and an optional S3-backed cluster model cache.

### Bug fixes

- Standardized gateway errors and health responses, preserved embedding timing headers, and fixed the scale-from-zero request path.
- Docling now reuses its converter and honors the selected device. PaddleOCR-VL generation now enables its key/value cache.
- Server wheels now include model and bundle configuration files, so installed packages can find their bundled defaults.

## v0.3.1 (2026-04-29)

Added BGE-M3 scoring with dense, sparse, ColBERT, and hybrid modes, plus Marqo e-commerce image-text embeddings. Failed model loads now enter an explicit failed state rather than remaining indefinitely in a loading state.

## v0.3.0 (2026-04-29)

### Highlights

Document extraction and multimodal retrieval expand substantially, while self-hosted clusters move to a Rust gateway with a dedicated configuration service.

### Features

- Added document inputs and structured extraction results across the server and SDKs, including Docling processing for PDF, DOCX, and HTML.
- Added GLM-OCR and PaddleOCR-VL-1.5; Qwen3-VL-Embedding-2B and Qwen3-VL-Reranker-2B; Qwen3-Reranker-0.6B and 4B; and SigLIP 2 image-text embeddings.
- Added GLiNER2, GLiNER-bi and Modern GLiNER-bi, Stablebridge token pruning and highlighting, and a ModernBERT-base embedding configuration.
- Added automatic GPU out-of-memory recovery and idle-model eviction, together with gateway and configuration-service metrics.

### Bug fixes

- Clients retry transient disconnects and capacity-related service errors without treating permanent connection failures as retryable.
- Improved propagation of model configuration changes and reporting of unknown or unroutable models.

### Breaking changes

- Self-hosted Helm configuration moves from `router` to `gateway` settings and adds a separate `config` service. Configuration writes belong to that service, not the inference gateway; review custom values and configuration clients when upgrading. The chart enables NATS JetStream queue routing by default.

## v0.2.0 (2026-04-17)

### Highlights

Added ModernBERT-based embedding models and LightOnOCR, along with startup model preloading and explicit concurrency control in the asynchronous Python client.

### Features

- Added GTE-ModernBERT-base, Snowflake Arctic Embed M v2.0, and IBM Granite English R2 embedding models, including the small variant.
- Added LightOnOCR-2-1B for OCR in the `transformers5` bundle.
- Added `max_concurrency` to `SIEAsyncClient` and Haystack-convention import aliases under `haystack_integrations`.
- Added anonymous usage telemetry, with opt-out through `SIE_TELEMETRY_DISABLED=true` or `DO_NOT_TRACK=1`.

### Bug fixes

- Model-affinity routing can spill requests to other workers instead of becoming stuck, and rejected requests now contribute to autoscaling demand.

### Breaking changes

- Worker startup no longer accepts `--model` to select models. Use `--preload` or `SIE_PRELOAD_MODELS` to load models at startup; otherwise models load on demand.

## v0.1.10 (2026-04-09)

### Highlights

Added LanceDB integrations for Python and TypeScript and a configuration-management API that distributes model changes to workers.

### Features

- Weaviate document enrichment now supports asynchronous processing, chunking, and streaming. LanceDB table enrichment processes batches incrementally without materializing the entire table.
- Added `get_model()` to the Python SDK and exposed queue-routing controls through Helm.

### Bug fixes

- Fixed queued score-response formatting, dead-letter routing, and reconnect handling.
- LlamaIndex embedding now handles `BytesIO` images, and Weaviate classification enrichment validates its configuration.

## v0.1.9 (2026-04-02)

Fixed Helm worker image tags to include the target platform and restored worker pool names to match machine profiles.

## v0.1.8 (2026-04-01)

Fixed duplicated platform suffixes in Helm worker image tags.

## v0.1.7 (2026-04-01)

### Highlights

Added Qdrant and Weaviate integrations and made self-hosted installation more complete through the Helm chart.

### Features

- Qdrant integration supports native sparse vectors; Weaviate integration supports the v4 client.
- ColBERT supports configurable document-length limits and custom prefix tokens.
- Pool creation accepts minimum worker counts and bundle selection. SDK/server version negotiation reports incompatible versions, and the Python SDK waits for capacity by default with a 900-second timeout.
- Helm now manages service accounts and model-access secrets, offers bundled autoscaling and monitoring components, and can pre-pull worker images onto GPU nodes.

### Bug fixes

- Corrected Qwen3 embedding attention behavior and LoRA-layer handling that could affect embedding results.
- Fixed asynchronous-client initialization outside a running event loop and extended pool leases to tolerate rolling upgrades.

### Breaking changes

- AWS and GCP Terraform deployments now separate cluster infrastructure from SIE application installation. Use Helm for the SIE application and review resource ownership before upgrading existing Terraform-managed installations.

## v0.1.6 (2026-03-12)

### Features

- Added Matryoshka embedding truncation for ColBERT.

### Bug fixes

- Fixed loading and configuration for NV-Embed-v2, Stella, BGE-M3, and instruction-based embedding models.
- Fixed ColBERT encoding on non-CUDA devices by selecting the native execution path.
- Aligned synchronous and asynchronous `encode()` and `score()` behavior. Malformed inference inputs now receive validation errors instead of server errors.
- Fixed spot-GPU resolution for autoscaling and increased default CPU-worker memory limits for the expanded bundle.

### Breaking changes

- The standalone `florence2` and `gliner` bundles are removed. Use the `default` bundle, which now includes Florence-2, GLiNER, GLiREL, and GLiClass dependencies.

## v0.1.5 (2026-02-27)

Added GLiNER v2.5 small, medium, and large models, GLiClass large models, and DeBERTa-based NLI classification. The gateway now streams request and response bodies, with corrected response headers for streaming.

## v0.1.4 (2026-02-27)

No user-facing changes.

## v0.1.3 (2026-02-26)

No user-facing changes.

## v0.1.2 (2026-02-26)

No user-facing changes.

## v0.1.1 (2026-02-26)

No user-facing changes.

## v0.1.0 (2026-02-26)

### Features

- Added structured gateway request logs and an `X-SIE-Worker` response header to identify the serving worker.

### Bug fixes

- Removed ColBERT's blanket CUDA-only model-loading restriction.
- GLiClass and NLI classification now populate `classifications` results correctly, and entity extraction handles typed dictionary results consistently.
- Corrected model-name resolution and bundle registration for classification models.

### Breaking changes

- Model configurations no longer accept per-model `dependencies`; adapter dependencies are defined by bundles. The `DEPENDENCY_CONFLICT` error and its HTTP 409 responses are removed. This does not remove HTTP 409 responses for incompatible bundle selections.
