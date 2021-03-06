# Traditional Gaussian without KSVD

This experiment is about compare traditional methods in relation to PSNR and SSIM metrics, on Gaussian noise, with mean 0 and std = 0.1.

Experiment started at 2021/02/24 23:06:08 and ended at 2021/02/24 23:07:32, during 83.88 seconds.

| Methods | Dataset | Training samples | Test samples | Dimension |
|---|---|---|---|---|
| BM3D,NLM,WST and Wiener Filter | BSD300 | 2400 | 600 | 50 X 50 |

# 1. Results

## 1.1 Table

We could see the results through the table, comparing all the methods in relation to PSNR and SSIM metrics.



| Method | PSNR (dB) | SSIM | Runtime (seconds) |
|---|---|---|---|
| BM3D | 27.73 ± 5.35 | 0.77 ± 0.2 | 62.96 |
| NLM | 26.42 ± 4.86 | 0.66 ± 0.26 | 18.86 |
| WST | 26.17 ± 5.15 | 0.76 ± 0.16 | 0.57 |
| Wiener Filter | 22.78 ± 3.42 | 0.81 ± 0.09 | 0.41 |
| Noisy | 21.39 ± 4.17 | 0.68 ± 0.18 | --- |

## 1.2 PSNR Boxplot

![PSNR boxplot](psnr_boxplot.png)

## 1.3 SSIM Boxplot

![SSIM boxplot](ssim_boxplot.png)


## 1.4 Visual Results

![Visual results](results.png)

## 1.5 DnCNN loss

![DnCNN loss](dncnn_loss.png)

## How to rerun the experiment?

If you want to rerun this experiment, you could use the `.metadata` directory.
This directory haves all the data, like the train and test data, the value of the PSNR methods in runtime, and data frame (pandas DataFrame) used to generate the Seaborn plots.