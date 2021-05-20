# DBT Alvarado 1

Comparing traditional and neural-network bas ed methods in relation to PSNR and SSIM metrics, on DBT projections.

Experiment started at 2021/05/19 20:23:32 and ended at 2021/05/19 20:39:25, during 00 hours 15 minutes and 52 seconds.


| Methods | Dataset | Training samples | Test samples | Dimension |
|---|---|---|---|---|
| BM3D,CGAN,CNN,Autoencoder,DnCNN,MLP,NLM,Wiener Filter and WST | dbt | 13200 | 3300 | 52 X 52 |

# 1. Results

## 1.1 Table

We could see the results through the table, comparing all the methods in relation to PSNR and SSIM metrics.



| Method | PSNR (dB) | SSIM | Runtime (seconds) |
|---|---|---|---|
| BM3D | 27.92 ± 2.5 | 0.74 ± 0.03 | 524.88 |
| CGAN | 29.31 ± 1.39 | 0.72 ± 0.03 | 41.74 |
| Wiener Filter | 20.7 ± 4.19 | 0.7 ± 0.03 | 2.21 |
| WST | 28.89 ± 1.29 | 0.67 ± 0.02 | 3.07 |
| Autoencoder | 25.1 ± 0.76 | 0.61 ± 0.07 | 8.25 |
| DnCNN | 23.06 ± 2.55 | 0.61 ± 0.08 | 202.09 |
| NLM | 25.62 ± 2.7 | 0.6 ± 0.06 | 112.01 |
| CNN | 23.18 ± 3.36 | 0.56 ± 0.09 | 39.72 |
| MLP | 20.19 ± 2.26 | 0.32 ± 0.04 | 6.3 |
| Noisy | 18.53 ± 3.62 | 0.35 ± 0.04 | --- |

## 1.2 PSNR Boxplot

![PSNR boxplot](psnr_boxplot.png)

## 1.3 SSIM Boxplot

![SSIM boxplot](ssim_boxplot.png)


## 1.4 Visual Results

![Visual results](results.png)

## 1.5 DnCNN loss

![DnCNN loss](DnCNN_loss.png)

## How to rerun the experiment?

If you want to rerun this experiment, you could use the `.metadata` directory.
This directory haves all the data, like the train and test data, the value of the PSNR methods in runtime, and data frame (pandas DataFrame) used to generate the Seaborn plots.