# Measured rehearsal results

Actual local run, September 17, 2026. These results describe this classroom classifier and workload only.

| Runtime | Accuracy | Macro-F1 | p50 | p95 | Weight size |
|---|---:|---:|---:|---:|---:|
| PyTorch FP32 | 90.25% | 0.9023 | 33.48 ms | 59.81 ms | 255.43 MiB |
| ONNX FP32 | 90.25% | 0.9023 | 30.33 ms | 41.56 ms | 255.53 MiB |
| ONNX INT8 | 90.50% | 0.9048 | 25.82 ms | 33.57 ms | 64.25 MiB |

Intel Core i9-14900HX, 4 CPU threads, batch 1, padded length 128  
100 sequential requests after 10 warm-ups  
Includes tokenization + forward. Excludes loading, network, and queueing.

Quality uses the same locked, balanced 800-example test set. Latency is one CPU run, not concurrent-load throughput. The 0.25 percentage-point accuracy difference does not establish that quantization improved quality. Weight size excludes configuration and tokenizer files.

Training used PEFT LoRA with 741,124 trainable parameters out of 67,697,672 in the wrapped model (1.09%). A complete base model remains necessary.

The real candidate passed the predeclared teaching gate. The deliberate failure separately reduces one class recall and should be rejected. Neither is a production certification.

Sources: `demo/results/evaluation.json`, `benchmark.json`, `training.json`, and `gate.json`. Raw timings, class reports, predictions, environment versions, and artifact identities are included.
