# Poisson-Gaussian Noise 2

Comparing traditional and neural-network based methods in relation to PSNR and SSIM metrics, on Poisson-Gaussian noise, with mean 0 and std = 0.1.

Experiment started at 2021/05/15 15:32:19 and ended at 2021/05/15 17:01:10, during 01 hours 28 minutes and 51 seconds.


| Methods | Dataset | Training samples | Test samples | Dimension |
|---|---|---|---|---|
| BM3D,CNN,Autoencoder,DnCNN,K-SVD,MLP,NLM,Wiener Filter and WST | BSD300 | 6000 | 1500 | 52 X 52 |

# 1. Results

## 1.1 Table

We could see the results through the table, comparing all the methods in relation to PSNR and SSIM metrics.



| Method | PSNR (dB) | SSIM | Runtime (seconds) |
|---|---|---|---|
| K-SVD | 23.82 ± 4.2 | 0.85 ± 0.06 | 5092.12 |
| Wiener Filter | 22.22 ± 3.81 | 0.83 ± 0.07 | 1.06 |
| BM3D | 26.16 ± 5.34 | 0.78 ± 0.17 | 122.08 |
| WST | 25.02 ± 4.9 | 0.78 ± 0.13 | 1.47 |
| DnCNN | 23.5 ± 5.06 | 0.75 ± 0.12 | 45.73 |
| NLM | 20.91 ± 3.83 | 0.68 ± 0.22 | 48.71 |
| CNN | 22.1 ± 5.45 | 0.62 ± 0.21 | 9.75 |
| Autoencoder | 25.68 ± 6.8 | 0.57 ± 0.28 | 1.95 |
| MLP | 8.32 ± 4.81 | 0.0 ± 0.02 | 1.97 |
| Noisy | 21.11 ± 4.27 | 0.7 ± 0.17 | --- |

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