# Poisson-Gaussian Noise 3

Comparing traditional and neural-network based methods in relation to PSNR and SSIM metrics, on Poisson-Gaussian noise, with mean 0 and std = 0.1.

Experiment started at 2021/05/15 20:57:56 and ended at 2021/05/15 22:25:58, during 01 hours 28 minutes and 01 seconds.


| Methods | Dataset | Training samples | Test samples | Dimension |
|---|---|---|---|---|
| BM3D,CNN,Autoencoder,DnCNN,K-SVD,MLP,NLM,Wiener Filter and WST | BSD300 | 6000 | 1500 | 52 X 52 |

# 1. Results

## 1.1 Table

We could see the results through the table, comparing all the methods in relation to PSNR and SSIM metrics.



| Method | PSNR (dB) | SSIM | Runtime (seconds) |
|---|---|---|---|
| K-SVD | 21.16 ± 4.87 | 0.83 ± 0.06 | 5041.93 |
| Wiener Filter | 21.03 ± 3.98 | 0.82 ± 0.07 | 1.06 |
| BM3D | 26.22 ± 5.05 | 0.78 ± 0.17 | 123.23 |
| WST | 24.45 ± 4.89 | 0.78 ± 0.13 | 1.47 |
| DnCNN | 22.24 ± 5.53 | 0.77 ± 0.1 | 45.55 |
| MLP | 22.59 ± 3.93 | 0.7 ± 0.18 | 1.91 |
| NLM | 20.35 ± 3.95 | 0.67 ± 0.22 | 48.67 |
| CNN | 21.28 ± 5.23 | 0.6 ± 0.22 | 9.49 |
| Autoencoder | 26.33 ± 7.52 | 0.57 ± 0.29 | 1.85 |
| Noisy | 20.71 ± 4.32 | 0.7 ± 0.17 | --- |

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