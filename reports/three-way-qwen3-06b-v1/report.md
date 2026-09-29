# Three-candidate validation comparison

Genuine validation predictions; test remains unused.

3100 requests: 3000 supported, 100 oos.

| Candidate | Raw macro-F1 | Invalid | Coverage | Routing error | Oos recall | Targets met |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| baseline | 0.882634 | 0 / 3100 | 0.684194 | 0.047619047619047616 | 0.9 | True |
| prompted | 0.272377 | 654 / 3100 | 0.000000 | None | 1.0 | False |
| finetuned | 0.966217 | 1 / 3100 | 0.803548 | 0.022480931352870333 | 0.9 | True |

## Which errors changed

supported: C fixed 2227 B errors and introduced 3 regressions; 101 remained wrong for both models.
oos: C fixed 16 B errors and introduced 28 regressions; 17 remained wrong for both models.

### fixed by c

- `5aa013df14bc2c8d6f1e7d90cf6a8f18728f42aa06aa83c24a6bd0923c727e76`: 'in french, how do i say, see you later'; gold `translate`, B `oos`, C `translate`.
- `782b066924fa03c1076a5a99aad2af5c0d087a924e6b142a8a79bf637d0e66d9`: 'how do you say hello in japanese'; gold `translate`, B `__invalid__`, C `translate`.
- `3460323f44897587ea23f3cc52615e6b80f0f7af8d213a980eaf9aeb1592e015`: 'how do i ask about the weather in chinese'; gold `translate`, B `__invalid__`, C `translate`.
- `c58a506c8e7d2b4146b48336764bc0e75e39f2225fda987926441bcfaf35bd87`: 'how can i say "cancel my order" in french'; gold `translate`, B `oos`, C `translate`.
- `ed84278be69a16d96ee5d60888ad52e5250644dea5bc414c4ce9a6e0f4285529`: 'how do i say dinner in spanish'; gold `translate`, B `__invalid__`, C `translate`.
- `d8d68c1d0a26226410ee425d5b5440d77d8ae4370d008384744d5985d6f3ea29`: 'how do you say good bye in french'; gold `translate`, B `goodbye`, C `translate`.
- `fe1341d2f5717db1c7874b1631b7d2a64428cff584b40967bea308b9a199fce3`: 'how do i say thank you in spanish'; gold `translate`, B `thank_you`, C `translate`.
- `8cb65b26c2e233f6a3c5d29c54e1f3b696dd9a56760db115163c6467406218ca`: 'how do i say good bye in chinese'; gold `translate`, B `goodbye`, C `translate`.
- `49f3e8ea47743a9cfb3d9ca2e00fba4a386a86a5e5c282a3d468c685b6eb6801`: 'how can i say thank you very much in chinese'; gold `translate`, B `thank_you`, C `translate`.
- `391cbc128c2e0a118f6c0a566f22d5b9322166c1aa539ae56d6179ea2ce83f27`: 'i need to know how to say hello in france'; gold `translate`, B `oos`, C `translate`.

### regressed in c

- `b45ebd913aaab87dbce31ef90f67c898d3f150986b9f0ffd6f55a2c317ec9e4e`: "i need to make a reservation to long horn's, can i"; gold `accept_reservations`, B `accept_reservations`, C `restaurant_reservation`.
- `c61c67b7ee5ece362763f043e4e6c2f3e43fab325ea9b98dca6c71ec250cfceb`: "there's something fishy on my card, report it"; gold `report_fraud`, B `report_fraud`, C `damaged_card`.
- `6ceee3cd7b53ae0e8a44a8a7ef5d6a2aa6e382e8ec0f5135c1d887252e22a758`: 'how much gas does it take to get to jackson'; gold `gas`, B `gas`, C `distance`.
- `85858b6f0a705427ddd0c673e4ac2958d487017e38f8fe17d64a7487dc41c5a7`: 'set a warning for when my bank account starts running low'; gold `oos`, B `oos`, C `account_blocked`.
- `3c3b2bab737931820042791b2fa5ee7ccfda94bffd7ea803f1d76ca3550196a0`: 'show me recent activity in my backyard'; gold `oos`, B `oos`, C `todo_list`.
- `7788c3de47a6c2cdb09155c6691d57088fcef5417c89c4d9fd42cf7d0aad7c08`: 'how long will it take me to pay off my card if i pay an extra $50 a month over the minimum'; gold `oos`, B `oos`, C `apr`.
- `131ed4234992d3c40d379b1f6207c82aba7dcb7ff85c87422e05a6922e8faafe`: 'can i mix antifreeze with water'; gold `oos`, B `oos`, C `__invalid__`.
- `3fe436ade282d7ee3e70b4785348467b474e81106ab1c386dc0fd5d6bb5736c3`: 'are any earning reports due'; gold `oos`, B `oos`, C `income`.
- `e118360c55143cf2b25049579e947434fd38e678240573a3995de55420ea168d`: 'how to unclog a drain'; gold `oos`, B `oos`, C `smart_home`.
- `02407b4c2375f1c917ec2956134660a1456371cf273343b1e00d22fdd410302b`: 'am kind of busy now'; gold `oos`, B `oos`, C `how_busy`.

All changed outputs are retained in changed_errors.jsonl. Examples above are deterministic and class-order biased. These are observed changes, not causal explanations or independent annotations.

Each policy uses the unchanged validation constraints and frozen A gate. Zero-route error is undefined. Invalid outputs and infrastructure failures remain in accounting. Exact counts, Wilson intervals, hardware, precision, timing scope and paired 1,000-resample bootstrap intervals (seed 42) are in comparison.json.

Checkpoint choice and these comparisons use validation, so estimates are subject to selection bias. Only 100 training oos examples are available; public benchmark pretraining contamination cannot be excluded. No test, deployment, cost-saving, or customer-performance claim is made.
