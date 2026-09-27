# Intervention v8.0 — Merged Results

Values are mean±sd across runs (single run → plain mean).
Gen-PPL columns are empty for `_shuffvar`-augmented shuffle rows (variance pass ran without generation).

## Band: early

| Model | Mode | N runs | EN acc | ZH acc | EN freePPL | ZH freePPL | EN genPPL | ZH genPPL |
|---|---|---|---|---|---|---|---|---|
| InternLM3-8B-Instruct | baseline | 1 | 85.0 | 100.0 | 43.81 | 71.07 | 1.52 | 1.57 |
| InternLM3-8B-Instruct | uniform_causal | 1 | 85.0 | 77.8 | 3283.00 | 13931.50 | 102.88 | 17.40 |
| InternLM3-8B-Instruct | uniform_global | 1 | 85.0 | 77.8 | 3389.16 | 14933.42 | 113.71 | 18.36 |
| InternLM3-8B-Instruct | shuffle_rows | 4 | 80.0±3.5 | 83.3±9.6 | 85.83±17.15 | 204.61±29.99 | 6.09 | 11.82 |
| Llama-3.1-8B | baseline | 1 | 85.0 | 100.0 | 43.15 | 21.76 | 1.86 | 2.00 |
| Llama-3.1-8B | uniform_causal | 1 | 85.0 | 90.0 | 212717.77 | 156006.59 | 701.05 | 1843.51 |
| Llama-3.1-8B | uniform_global | 1 | 85.0 | 90.0 | 5837544.93 | 136621.15 | 113.34 | 276.35 |
| Llama-3.1-8B | shuffle_rows | 4 | 87.5±2.5 | 82.5±4.3 | 116.10±7.52 | 302.37±46.56 | 62.96 | 62.61 |
| Llama-3.1-8B-Base | baseline | 1 | 85.0 | 100.0 | 34.35 | 25.03 | 2.31 | 1.92 |
| Llama-3.1-8B-Base | uniform_causal | 1 | 90.0 | 90.9 | 6550.35 | 18620.33 | 2324.95 | 590.15 |
| Llama-3.1-8B-Base | uniform_global | 1 | 85.0 | 90.9 | 7357.16 | 6463.06 | 2982.40 | 324.31 |
| Llama-3.1-8B-Base | shuffle_rows | 4 | 86.2±2.2 | 88.6±7.5 | 73.44±14.19 | 378.23±57.83 | 17.44 | 23.84 |
| Mistral-7B | baseline | 1 | 90.0 | 100.0 | 39.42 | 13.51 | 1.83 | 1.79 |
| Mistral-7B | uniform_causal | 1 | 90.0 | 76.9 | 182.27 | 758.58 | 11.23 | 37.43 |
| Mistral-7B | uniform_global | 1 | 90.0 | 69.2 | 142.12 | 504.70 | 7.31 | 88.62 |
| Mistral-7B | shuffle_rows | 4 | 81.2±2.2 | 75.0±6.4 | 101.10±35.50 | 215.39±29.64 | 25.03 | 31.28 |
| Mixtral-8x7B-Instruct | baseline | 1 | 90.0 | 100.0 | 48.58 | 12.14 | 1.98 | 1.63 |
| Mixtral-8x7B-Instruct | uniform_causal | 1 | 90.0 | 90.0 | 813.66 | 1067.10 | 51.61 | 441.14 |
| Mixtral-8x7B-Instruct | uniform_global | 1 | 85.0 | 90.0 | 453.32 | 1028.67 | 77.47 | 117.34 |
| Mixtral-8x7B-Instruct | shuffle_rows | 4 | 88.8±2.2 | 67.5±20.5 | 217.88±89.37 | 303.11±9.00 | 62.08 | 40.12 |
| Qwen3-0.6B | baseline | 1 | 90.0 | 100.0 | 71.68 | 204.54 | 1.79 | 2.37 |
| Qwen3-0.6B | uniform_causal | 1 | 90.0 | 100.0 | 2174221.61 | 23013904.75 | 140.15 | 99.79 |
| Qwen3-0.6B | uniform_global | 1 | 90.0 | 90.0 | 7866348.12 | 8891141.52 | 107.96 | 109.78 |
| Qwen3-0.6B | shuffle_rows | 4 | 86.2±2.2 | 90.0±7.1 | 30380.66±12814.90 | 38920.91±9591.93 | 261.49 | 299.46 |
| Qwen3-1.7B | baseline | 1 | 90.0 | 100.0 | 27.98 | 72.32 | 1.58 | 1.63 |
| Qwen3-1.7B | uniform_causal | 1 | 85.0 | 90.0 | 229002.01 | 89443.72 | 18.53 | 26.30 |
| Qwen3-1.7B | uniform_global | 1 | 90.0 | 90.0 | 281023.67 | 3270734.22 | 10.17 | 9.14 |
| Qwen3-1.7B | shuffle_rows | 4 | 85.0±0.0 | 90.0±0.0 | 1095722.10±1057512.60 | 274293.09±121656.83 | 1856.60 | 1590.52 |
| Qwen3-14B | baseline | 1 | 90.0 | 100.0 | 28.20 | 40.85 | 1.33 | 1.53 |
| Qwen3-14B | uniform_causal | 1 | 85.0 | 77.8 | 24940.48 | 78130.75 | 403.82 | 407.60 |
| Qwen3-14B | uniform_global | 1 | 85.0 | 66.7 | 24217.44 | 288039.22 | 184.78 | 206.07 |
| Qwen3-14B | shuffle_rows | 4 | 85.0±0.0 | 80.6±4.8 | 5188.40±1276.87 | 30545.20±30464.55 | 213.72 | 694.75 |
| Qwen3-4B | baseline | 1 | 90.0 | 100.0 | 27.16 | 48.52 | 1.35 | 1.52 |
| Qwen3-4B | uniform_causal | 1 | 90.0 | 100.0 | 34209.17 | 39046.45 | 1371.65 | 337.68 |
| Qwen3-4B | uniform_global | 1 | 85.0 | 88.9 | 92949.00 | 78707.97 | 124.49 | 59.56 |
| Qwen3-4B | shuffle_rows | 4 | 87.5±2.5 | 80.6±4.8 | 17172.74±7596.26 | 102903.03±91035.59 | 208.27 | 436.41 |
| Qwen3-8B | baseline | 1 | 90.0 | 100.0 | 19.43 | 42.09 | 1.24 | 1.59 |
| Qwen3-8B | uniform_causal | 1 | 90.0 | 90.0 | 742.27 | 3211.88 | 99.78 | 57.21 |
| Qwen3-8B | uniform_global | 1 | 85.0 | 90.0 | 1372.52 | 28563.93 | 31.82 | 28.36 |
| Qwen3-8B | shuffle_rows | 4 | 87.5±2.5 | 90.0±7.1 | 5029.86±1434.91 | 486100.26±464460.26 | 43.34 | 261.07 |
| Qwen3-8B-Base | baseline | 1 | 90.0 | 100.0 | 15.20 | 22.32 | 1.65 | 1.79 |
| Qwen3-8B-Base | uniform_causal | 1 | 85.0 | 90.9 | 478.36 | 1906.16 | 20.21 | 131.38 |
| Qwen3-8B-Base | uniform_global | 1 | 85.0 | 90.9 | 900.90 | 8176.22 | 56.04 | 845.34 |
| Qwen3-8B-Base | shuffle_rows | 4 | 86.2±2.2 | 86.4±7.9 | 1596.41±510.91 | 3218.65±980.72 | 162.99 | 85.22 |

## Band: mid

| Model | Mode | N runs | EN acc | ZH acc | EN freePPL | ZH freePPL | EN genPPL | ZH genPPL |
|---|---|---|---|---|---|---|---|---|
| InternLM3-8B-Instruct | baseline | 1 | 85.0 | 100.0 | 43.81 | 71.07 | 1.52 | 1.57 |
| InternLM3-8B-Instruct | uniform_causal | 1 | 85.0 | 77.8 | 105.12 | 234.75 | 1.62 | 1.81 |
| InternLM3-8B-Instruct | uniform_global | 1 | 85.0 | 88.9 | 20.69 | 34.20 | 1.56 | 2.07 |
| InternLM3-8B-Instruct | shuffle_rows | 1 | 85.0 | 100.0 | 43.40 | 83.04 | 10.34 | 29.21 |
| Llama-3.1-8B | baseline | 1 | 85.0 | 100.0 | 43.15 | 21.76 | 1.86 | 2.00 |
| Llama-3.1-8B | uniform_causal | 1 | 90.0 | 100.0 | 242.59 | 823.78 | 1.45 | 2.25 |
| Llama-3.1-8B | uniform_global | 1 | 85.0 | 90.0 | 30.79 | 1404.00 | 1.20 | 1.69 |
| Llama-3.1-8B | shuffle_rows | 1 | 90.0 | 100.0 | 53.81 | 33.65 | 10.05 | 15.14 |
| Llama-3.1-8B-Base | baseline | 1 | 85.0 | 100.0 | 34.35 | 25.03 | 2.31 | 1.92 |
| Llama-3.1-8B-Base | uniform_causal | 1 | 90.0 | 100.0 | 123.34 | 7085.35 | 1.29 | 1.61 |
| Llama-3.1-8B-Base | uniform_global | 1 | 85.0 | 72.7 | 12.70 | 270714.66 | 1.13 | 1.81 |
| Llama-3.1-8B-Base | shuffle_rows | 1 | 90.0 | 90.9 | 32.64 | 40.41 | 8.47 | 16.27 |
| Mistral-7B | baseline | 1 | 90.0 | 100.0 | 39.42 | 13.51 | 1.83 | 1.79 |
| Mistral-7B | uniform_causal | 1 | 90.0 | 100.0 | 49.62 | 67.03 | 2.09 | 1.75 |
| Mistral-7B | uniform_global | 1 | 85.0 | 92.3 | 11.18 | 25.81 | 1.89 | 5.18 |
| Mistral-7B | shuffle_rows | 1 | 90.0 | 100.0 | 37.62 | 20.41 | 6.88 | 7.85 |
| Mixtral-8x7B-Instruct | baseline | 1 | 90.0 | 100.0 | 48.58 | 12.14 | 1.98 | 1.63 |
| Mixtral-8x7B-Instruct | uniform_causal | 1 | 90.0 | 100.0 | 59.06 | 26.26 | 3.08 | 5.22 |
| Mixtral-8x7B-Instruct | uniform_global | 1 | 90.0 | 100.0 | 7.73 | 7.42 | 2.08 | 3.45 |
| Mixtral-8x7B-Instruct | shuffle_rows | 1 | 90.0 | 100.0 | 55.09 | 27.87 | 7.96 | 7.26 |
| Qwen3-0.6B | baseline | 1 | 90.0 | 100.0 | 71.68 | 204.54 | 1.79 | 2.37 |
| Qwen3-0.6B | uniform_causal | 1 | 90.0 | 90.0 | 1069.29 | 538.07 | 1.49 | 1.25 |
| Qwen3-0.6B | uniform_global | 1 | 90.0 | 90.0 | 708.62 | 2539.90 | 1.52 | 1.61 |
| Qwen3-0.6B | shuffle_rows | 1 | 85.0 | 90.0 | 235.96 | 668.05 | 77.22 | 88.13 |
| Qwen3-1.7B | baseline | 1 | 90.0 | 100.0 | 27.98 | 72.32 | 1.58 | 1.63 |
| Qwen3-1.7B | uniform_causal | 1 | 90.0 | 90.0 | 215.75 | 167.00 | 1.49 | 1.80 |
| Qwen3-1.7B | uniform_global | 1 | 85.0 | 90.0 | 13.40 | 31.12 | 1.01 | 1.66 |
| Qwen3-1.7B | shuffle_rows | 1 | 85.0 | 90.0 | 54.20 | 103.02 | 14.13 | 33.73 |
| Qwen3-14B | baseline | 1 | 90.0 | 100.0 | 28.20 | 40.85 | 1.33 | 1.53 |
| Qwen3-14B | uniform_causal | 1 | 90.0 | 88.9 | 263.52 | 220.71 | 1.24 | 1.20 |
| Qwen3-14B | uniform_global | 1 | 90.0 | 88.9 | 18.45 | 52.24 | 1.06 | 1.10 |
| Qwen3-14B | shuffle_rows | 1 | 90.0 | 88.9 | 128.56 | 139.43 | 11.32 | 17.99 |
| Qwen3-4B | baseline | 1 | 90.0 | 100.0 | 27.16 | 48.52 | 1.35 | 1.52 |
| Qwen3-4B | uniform_causal | 1 | 90.0 | 100.0 | 162.18 | 355.30 | 1.39 | 1.30 |
| Qwen3-4B | uniform_global | 1 | 90.0 | 88.9 | 31.31 | 45.28 | 1.07 | 1.12 |
| Qwen3-4B | shuffle_rows | 1 | 90.0 | 100.0 | 96.83 | 278.88 | 30.11 | 56.28 |
| Qwen3-8B | baseline | 1 | 90.0 | 100.0 | 19.43 | 42.09 | 1.24 | 1.59 |
| Qwen3-8B | uniform_causal | 1 | 90.0 | 100.0 | 174.73 | 143.87 | 1.33 | 1.23 |
| Qwen3-8B | uniform_global | 1 | 90.0 | 100.0 | 8.30 | 18.85 | 1.07 | 1.07 |
| Qwen3-8B | shuffle_rows | 1 | 90.0 | 100.0 | 221.35 | 235.46 | 22.43 | 37.70 |
| Qwen3-8B-Base | baseline | 1 | 90.0 | 100.0 | 15.20 | 22.32 | 1.65 | 1.79 |
| Qwen3-8B-Base | uniform_causal | 1 | 90.0 | 100.0 | 65.31 | 56.89 | 1.23 | 1.38 |
| Qwen3-8B-Base | uniform_global | 1 | 90.0 | 90.9 | 6.09 | 11.79 | 1.14 | 1.21 |
| Qwen3-8B-Base | shuffle_rows | 1 | 90.0 | 100.0 | 50.80 | 82.86 | 16.09 | 25.41 |

## Band: late

| Model | Mode | N runs | EN acc | ZH acc | EN freePPL | ZH freePPL | EN genPPL | ZH genPPL |
|---|---|---|---|---|---|---|---|---|
| InternLM3-8B-Instruct | baseline | 1 | 85.0 | 100.0 | 43.81 | 71.07 | 1.52 | 1.57 |
| InternLM3-8B-Instruct | uniform_causal | 1 | 85.0 | 100.0 | 60.65 | 90.66 | 2.71 | 3.39 |
| InternLM3-8B-Instruct | uniform_global | 1 | 85.0 | 88.9 | 22.16 | 20.48 | 3.00 | 3.33 |
| InternLM3-8B-Instruct | shuffle_rows | 1 | 85.0 | 100.0 | 31.84 | 41.87 | 1.99 | 2.06 |
| Llama-3.1-8B | baseline | 1 | 85.0 | 100.0 | 43.15 | 21.76 | 1.86 | 2.00 |
| Llama-3.1-8B | uniform_causal | 1 | 85.0 | 100.0 | 63.83 | 83.75 | 2.47 | 7.27 |
| Llama-3.1-8B | uniform_global | 1 | 85.0 | 100.0 | 69.07 | 101.61 | 1.79 | 5.63 |
| Llama-3.1-8B | shuffle_rows | 1 | 90.0 | 100.0 | 40.37 | 24.37 | 2.72 | 2.75 |
| Llama-3.1-8B-Base | baseline | 1 | 85.0 | 100.0 | 34.35 | 25.03 | 2.31 | 1.92 |
| Llama-3.1-8B-Base | uniform_causal | 1 | 85.0 | 100.0 | 43.05 | 77.32 | 4.33 | 4.88 |
| Llama-3.1-8B-Base | uniform_global | 1 | 85.0 | 100.0 | 38.11 | 70.86 | 3.22 | 4.01 |
| Llama-3.1-8B-Base | shuffle_rows | 1 | 85.0 | 100.0 | 33.18 | 28.96 | 2.65 | 2.35 |
| Mistral-7B | baseline | 1 | 90.0 | 100.0 | 39.42 | 13.51 | 1.83 | 1.79 |
| Mistral-7B | uniform_causal | 1 | 90.0 | 92.3 | 45.57 | 25.29 | 2.87 | 2.64 |
| Mistral-7B | uniform_global | 1 | 90.0 | 84.6 | 40.81 | 17.54 | 2.56 | 2.38 |
| Mistral-7B | shuffle_rows | 1 | 90.0 | 92.3 | 43.59 | 12.86 | 2.88 | 2.53 |
| Mixtral-8x7B-Instruct | baseline | 1 | 90.0 | 100.0 | 48.58 | 12.14 | 1.98 | 1.63 |
| Mixtral-8x7B-Instruct | uniform_causal | 1 | 90.0 | 100.0 | 54.23 | 16.24 | 6.19 | 3.25 |
| Mixtral-8x7B-Instruct | uniform_global | 1 | 90.0 | 100.0 | 21.97 | 9.73 | 3.97 | 2.95 |
| Mixtral-8x7B-Instruct | shuffle_rows | 1 | 90.0 | 100.0 | 47.73 | 16.08 | 3.77 | 3.41 |
| Qwen3-0.6B | baseline | 1 | 90.0 | 100.0 | 71.68 | 204.54 | 1.79 | 2.37 |
| Qwen3-0.6B | uniform_causal | 1 | 90.0 | 100.0 | 279.31 | 498.45 | 327.87 | 11.87 |
| Qwen3-0.6B | uniform_global | 1 | 90.0 | 100.0 | 53.59 | 81.92 | 13.33 | 6.51 |
| Qwen3-0.6B | shuffle_rows | 1 | 90.0 | 100.0 | 60.23 | 165.03 | 4.14 | 5.43 |
| Qwen3-1.7B | baseline | 1 | 90.0 | 100.0 | 27.98 | 72.32 | 1.58 | 1.63 |
| Qwen3-1.7B | uniform_causal | 1 | 85.0 | 100.0 | 55.64 | 91.51 | 2.86 | 7.04 |
| Qwen3-1.7B | uniform_global | 1 | 85.0 | 90.0 | 24.17 | 19.16 | 2.39 | 5.83 |
| Qwen3-1.7B | shuffle_rows | 1 | 90.0 | 100.0 | 23.81 | 50.67 | 2.52 | 3.12 |
| Qwen3-14B | baseline | 1 | 90.0 | 100.0 | 28.20 | 40.85 | 1.33 | 1.53 |
| Qwen3-14B | uniform_causal | 1 | 90.0 | 88.9 | 90.81 | 140.88 | 2.00 | 13.20 |
| Qwen3-14B | uniform_global | 1 | 85.0 | 88.9 | 33.83 | 61.25 | 2.51 | 331.79 |
| Qwen3-14B | shuffle_rows | 1 | 85.0 | 88.9 | 59.78 | 91.13 | 4.09 | 4.17 |
| Qwen3-4B | baseline | 1 | 90.0 | 100.0 | 27.16 | 48.52 | 1.35 | 1.52 |
| Qwen3-4B | uniform_causal | 1 | 85.0 | 100.0 | 46.70 | 85.42 | 189.46 | 6.15 |
| Qwen3-4B | uniform_global | 1 | 85.0 | 88.9 | 18.89 | 43.26 | 21.44 | 6.61 |
| Qwen3-4B | shuffle_rows | 1 | 90.0 | 100.0 | 24.31 | 57.77 | 3.80 | 4.66 |
| Qwen3-8B | baseline | 1 | 90.0 | 100.0 | 19.43 | 42.09 | 1.24 | 1.59 |
| Qwen3-8B | uniform_causal | 1 | 85.0 | 100.0 | 29.42 | 53.40 | 5.10 | 11.12 |
| Qwen3-8B | uniform_global | 1 | 85.0 | 100.0 | 11.28 | 19.77 | 3.21 | 6.16 |
| Qwen3-8B | shuffle_rows | 1 | 90.0 | 100.0 | 20.70 | 41.59 | 7.54 | 3.82 |
| Qwen3-8B-Base | baseline | 1 | 90.0 | 100.0 | 15.20 | 22.32 | 1.65 | 1.79 |
| Qwen3-8B-Base | uniform_causal | 1 | 85.0 | 100.0 | 17.69 | 32.40 | 4.17 | 4.79 |
| Qwen3-8B-Base | uniform_global | 1 | 85.0 | 100.0 | 10.44 | 18.21 | 4.02 | 3.99 |
| Qwen3-8B-Base | shuffle_rows | 1 | 90.0 | 100.0 | 13.67 | 20.78 | 3.11 | 2.86 |
