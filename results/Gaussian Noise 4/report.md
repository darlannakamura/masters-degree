# Gaussian Noise 4

Comparing traditional and neural-network based methods in relation to PSNR and SSIM metrics, on Gaussian noise, with mean 0 and std = 0.1.

Experiment started at 2021/05/06 18:06:35 and ended at 2021/05/06 19:32:31, during 01 hours 25 minutes and 56 seconds.


| Methods | Dataset | Training samples | Test samples | Dimension |
|---|---|---|---|---|
| BM3D,CNN,Autoencoder,DnCNN,K-SVD,MLP,NLM,Wiener Filter and WST | BSD300 | 6000 | 1500 | 52 X 52 |

# 1. Results

## 1.1 Table

We could see the results through the table, comparing all the methods in relation to PSNR and SSIM metrics.



| Method | PSNR (dB) | SSIM | Runtime (seconds) |
|---|---|---|---|
| K-SVD | 22.14 ± 4.29 | 0.84 ± 0.06 | 4921.15 |
| Wiener Filter | 22.14 ± 3.81 | 0.83 ± 0.07 | 1.02 |
| DnCNN | 20.69 ± 5.25 | 0.78 ± 0.08 | 44.81 |
| WST | 24.75 ± 5.06 | 0.78 ± 0.13 | 1.45 |
| BM3D | 25.7 ± 5.32 | 0.77 ± 0.17 | 120.74 |
| NLM | 21.74 ± 3.9 | 0.68 ± 0.22 | 48.15 |
| CNN | 21.66 ± 4.57 | 0.62 ± 0.2 | 9.38 |
| Autoencoder | 25.52 ± 6.57 | 0.56 ± 0.28 | 1.95 |
| MLP | 8.32 ± 4.81 | 0.0 ± 0.02 | 1.92 |
| Noisy | 20.18 ± 4.37 | 0.7 ± 0.16 | --- |

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