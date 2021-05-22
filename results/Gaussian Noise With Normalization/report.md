# Gaussian Noise With Normalization

Comparing traditional and neural-network based methods in relation to PSNR and SSIM metrics, on Gaussian noise, with mean 0 and std = 0.1.

Experiment started at 2021/05/21 14:31:03 and ended at 2021/05/22 04:20:19, during 13 hours 49 minutes and 15 seconds.


| Methods | Dataset | Training samples | Test samples | Dimension |
|---|---|---|---|---|
| BM3D,CGAN,CNN,DIP,DnCNN,DnCNN10,K-SVD,MLP,NLM,Wiener Filter and WST | BSD300 | 24000 | 6000 | 50 X 50 |

# 1. Results

## 1.1 Table

We could see the results through the table, comparing all the methods in relation to PSNR and SSIM metrics.



| Method | PSNR (dB) | SSIM | Runtime (seconds) |
|---|---|---|---|
| DnCNN10 | 25.76 ± 5.02 | 0.9 ± 0.06 | 79.28 |
| DnCNN | 25.71 ± 4.73 | 0.87 ± 0.05 | 166.22 |
| K-SVD | 21.97 ± 4.28 | 0.83 ± 0.06 | 17493.5 |
| Wiener Filter | 20.78 ± 3.52 | 0.79 ± 0.09 | 4.01 |
| WST | 24.18 ± 4.68 | 0.73 ± 0.15 | 5.65 |
| BM3D | 22.45 ± 4.67 | 0.71 ± 0.2 | 507.85 |
| CGAN | 15.56 ± 3.29 | 0.7 ± 0.14 | 38.04 |
| CNN | 20.77 ± 4.35 | 0.7 ± 0.15 | 30.9 |
| MLP | 21.79 ± 3.15 | 0.69 ± 0.18 | 6.23 |
| NLM | 24.15 ± 3.93 | 0.62 ± 0.25 | 185.32 |
| DIP | 18.43 ± 3.56 | 0.61 ± 0.24 | 31210.8 |
| Noisy | 19.58 ± 4.07 | 0.7 ± 0.16 | --- |

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