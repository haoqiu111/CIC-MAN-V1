# v2 Shortcut Reversal (target=paderborn task, seed 42)

Class-correlated 35/55/75 Hz tone injected into source recordings; target tone correlated/reversed/neutral on the fixed seed-42 subset (n=300 recordings).

| Model | Corr Rec-F1 | Reversed Rec-F1 | Neutral Rec-F1 | Reversal Gap |
|---|---:|---:|---:|---:|
| single_raw | 0.5517 | 0.0594 | 0.1061 | 0.4924 |
| dann | 0.6413 | 0.1357 | 0.1642 | 0.5056 |
| dg_irm | 0.7146 | 0.1842 | 0.1606 | 0.5305 |
| dg_coral | 0.7314 | 0.1946 | 0.1835 | 0.5368 |
| dg_mmd | 0.6822 | 0.1734 | 0.1862 | 0.5088 |
| dg_groupdro | 0.6501 | 0.1406 | 0.1922 | 0.5095 |
| moe | 0.9284 | 0.3106 | 0.1854 | 0.6178 |
| ensemble | 0.7154 | 0.1842 | 0.1790 | 0.5313 |
| cicman_v4 | 1.0000 | 0.3333 | 0.1096 | 0.6667 |
| single_env_order | 0.3555 | 0.3555 | 0.3555 | 0.0000 |
| cicman_v6ic | 0.8624 | 0.3475 | 0.2612 | 0.5149 |
