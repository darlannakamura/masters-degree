# DBT Alvarado 1

Comparing traditional and neural-network bas ed methods in relation to PSNR and SSIM metrics, on DBT projections.

Experiment started at 2021/05/18 21:53:01 and ended at 2021/05/18 21:55:38, during 00 hours 02 minutes and 36 seconds.


| Methods | Dataset | Training samples | Test samples | Dimension |
|---|---|---|---|---|
| CGAN,CNN,Autoencoder,DnCNN,MLP,NLM,Wiener Filter and WST | dbt | 6600 | 1650 | 52 X 52 |

# 1. Results

## 1.1 Table

We could see the results through the table, comparing all the methods in relation to PSNR and SSIM metrics.



| Method | PSNR (dB) | SSIM | Runtime (seconds) |
|---|---|---|---|
| BM3D | 28.02 ± 2.65 | 0.73 ± 0.03 | 139.21 |
| Wiener Filter | 20.83 ± 4.34 | 0.7 ± 0.02 | 1.11 |
| DIP | 21.35 ± 4.98 | 0.66 ± 0.05 | 9670.17 |
| K-SVD | 22.28 ± 4.0 | 0.66 ± 0.03 | 5645.98 |
| WST | 24.53 ± 2.64 | 0.63 ± 0.02 | 1.6 |
| NLM | 25.36 ± 2.91 | 0.58 ± 0.07 | 54.49 |
| DnCNN | 14.21 ± 4.85 | 0.47 ± 0.09 | 49.23 |
| CGAN | 10.25 ± 5.89 | 0.31 ± 0.05 | 18.44 |
| CNN | 13.19 ± 2.59 | 0.06 ± 0.01 | 21.9 |
| Autoencoder | 5.02 ± 1.16 | 0.0 ± 0.0 | 1.94 |
| MLP | 4.78 ± 3.57 | 0.0 ± 0.0 | 2.21 |
| Noisy | 19.38 ± 3.23 | 0.34 ± 0.03 | --- |

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