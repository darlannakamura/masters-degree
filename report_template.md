# $EXPERIMENT_NAME

$ABOUT

Experiment started at $START_DATE and ended at $END_DATE, during $DURATION.


| Methods | Dataset | Training samples | Test samples | Dimension |
|---|---|---|---|---|
| $METHODS | $DATASET | $TRAINING_SAMPLES | $TEST_SAMPLES | $DIMENSION |

# 1. Results

## 1.1 Table

We could see the results through the table, comparing all the methods in relation to PSNR and SSIM metrics.

$TABLE

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