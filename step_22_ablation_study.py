"""## 22. Ablation study — T4 GPU

For the final project, rerun SCORE-BN with these settings and record each result:

| Run | `lambda_cons` | `lambda_under` |
|---|---:|---:|
| Standard classifier | 0.0 | 0.0 |
| Consistency only | 0.5 | 0.0 |
| Risk only | 0.0 | 0.3 |
| Full SCORE-BN | 0.5 | 0.3 |

For a defensible conference extension, run at least five random seeds and bootstrap confidence intervals. The training cell can be placed inside a function and called for each configuration.

