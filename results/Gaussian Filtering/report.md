# Gaussian Filtering

This experiment is about compare traditional methods in relation to PSNR and SSIM metrics, on Gaussian noise, with mean 0 and std = 0.1.

Experiment started at 2021/02/24 00:20:46 and ended at 2021/02/24 00:59:20, during 2313.69 seconds.

| Methods | Dataset | Training samples | Test samples | Dimension |
|---|---|---|---|---|
| DnCNN,K-SVD,BM3D,NLM,WST and Wiener Filter | BSD300 | 2400 | 600 | 50 X 50 |

# 1. Results

## 1.1 Table

We could see the results through the table, comparing all the methods in relation to PSNR and SSIM metrics.



| Method | PSNR (dB) | SSIM | Runtime (seconds) |
|---|---|---|---|
| DnCNN | 18.93 ± 2.81 | 0.7 ± 0.12 | 21.67 |
| K-SVD | 25.02 ± 1.03 | 0.52 ± 0.17 | 2212.06 |
| BM3D | 25.35 ± 4.37 | 0.8 ± 0.15 | 59.05 |
| NLM | 30.14 ± 6.22 | 0.76 ± 0.2 | 18.47 |
| WST | 26.9 ± 2.99 | 0.68 ± 0.08 | 0.54 |
| Wiener Filter | 26.18 ± 1.58 | 0.58 ± 0.13 | 0.41 |
| Noisy | 20.33 ± 0.61 | 0.3 ± 0.2 | --- |

## 1.2 PSNR Boxplot

![PSNR boxplot](psnr_boxplot.png)

## 1.3 SSIM Boxplot

![SSIM boxplot](ssim_boxplot.png)

## How to rerun the experiment?

If you want to rerun this experiment, you could use the `.metadata` directory.
This directory haves all the data, like the train and test data, the value of the PSNR methods in runtime, and data frame (pandas DataFrame) used to generate the Seaborn plots.