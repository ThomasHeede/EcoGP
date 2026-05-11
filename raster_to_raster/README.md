# EcoGP: Raster to Raster

This package generates raster files for species predictions from a directory of environmental raster files, in three steps. The pretrained model must have been trained on the exact same variables as those present in the input directory.

---

## Step 0: Set Up Requirements

Before running the code, set up a conda environment:

1. Create the environment:
   `conda env create -f r2r.yml`

2. Activate the environment:
   `conda activate r2r`

3. Navigate to the parent directory:
   `cd Your/Path/To/EcoGP/raster_to_raster/` *(adjust to match your local path)*

---

## Step 1: Generate Feature Table

This step stacks the individual feature rasters and flattens them into a vertical table:

| **Pixel \ Feature** | **A** | **B** | **C** | $\cdots$ |
|:-------------------:|-------|-------|-------|----------|
|        **1**        |       |       |       |          |
|        **2**        |       |       |       |          |
|      $\vdots$       |       |       |       |          |

Six variables can be configured. We recommend running the script first with the provided test data (no changes needed) to verify everything works on your machine.

Once confirmed, you can remove the `"test/"` prefix from the paths — everything else is modifiable.

```bash
INPUT_RASTER_DIR="test/data/input_raster/"  # Folder in which rasters can be found – has to match 100% with pretrained model!
OUTPUT_RASTER_DIR="test/data/output_raster/"  # Folder in which rasters with species predictions should be placed.
INTERMEDIATE_DIR="test/data/model_data/"  # Saving intermediate data in this folder.
PRETRAINED_MODEL_DIR="test/pretrained_model/"  # Location for pretrained model, parameters and data handling similar to training data.
N_SAMPLES=100  # Number of samples to derive species predictions.
BATCH_SIZE=512  # Number in each batch for optimization.
```

Run the first script to generate the table above, along with pixel coordinates:

```bash
python rasters_to_csv.py \
  --input-dir "$INPUT_RASTER_DIR" \
  --output-dir "$INTERMEDIATE_DIR"
```

---

## Step 2: Run Pretrained Prediction Module

This produces a CSV with species predictions for each pixel:

```bash
python preds_from_pretrained.py \
  --input-dir "$INTERMEDIATE_DIR" \
  --output-dir "$INTERMEDIATE_DIR" \
  --pretrained-dir "$PRETRAINED_MODEL_DIR" \
  --n-samples "$N_SAMPLES" \
  --batch-size "$BATCH_SIZE" 
```

---

## Step 3: Generate Species Rasters

This converts the predictions into raster files, using the first `.tif` file as a spatial reference:

```bash
python csv_to_rasters.py \
  --prediction-dir "$INTERMEDIATE_DIR" \
  --reference-tif $INPUT_RASTER_DIR \
  --output-dir "$OUTPUT_RASTER_DIR" 
```
