# Poisson Noise 1

Comparing traditional and neural-network bas ed methods in relation to PSNR and SSIM metrics, on Poisson noise.

Experiment started at 2021/05/07 01:45:38 and ended at 2021/05/07 14:55:17, during 13 hours 09 minutes and 39 seconds.


| Methods | Dataset | Training samples | Test samples | Dimension |
|---|---|---|---|---|
| BM3D,CNN,Autoencoder,DIP,DnCNN,K-SVD,MLP,NLM,Wiener Filter and WST | BSD300 | 6000 | 1500 | 52 X 52 |

# 1. Results

## 1.1 Table

We could see the results through the table, comparing all the methods in relation to PSNR and SSIM metrics.



| Method | PSNR (dB) | SSIM | Runtime (seconds) |
|---|---|---|---|
| K-SVD | 23.59 ± 4.96 | 0.84 ± 0.06 | 5310.24 |
| Wiener Filter | 21.07 ± 3.95 | 0.82 ± 0.07 | 1.05 |
| DIP | 21.98 ± 4.49 | 0.8 ± 0.13 | 41787.89 |
| WST | 24.93 ± 4.7 | 0.79 ± 0.13 | 1.49 |
| BM3D | 24.85 ± 5.1 | 0.78 ± 0.17 | 144.43 |
| DnCNN | 23.18 ± 4.79 | 0.78 ± 0.09 | 58.5 |
| MLP | 22.05 ± 4.06 | 0.7 ± 0.17 | 2.36 |
| NLM | 21.74 ± 3.89 | 0.68 ± 0.22 | 48.23 |
| CNN | 22.76 ± 5.23 | 0.62 ± 0.2 | 14.84 |
| Autoencoder | 25.45 ± 6.59 | 0.57 ± 0.28 | 2.81 |
| Noisy | 20.26 ± 4.37 | 0.7 ± 0.16 | --- |

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