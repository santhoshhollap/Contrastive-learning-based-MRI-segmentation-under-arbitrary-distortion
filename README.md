# Contrastive-learning-based-MRI-segmentation-under-arbitrary-distortion

# Pipeline Setup Instructions

## Steps to Setup the Pipeline

1. **Clone the Repository**
   - Download the Git repository to your personal system.

2. **Generate Distorted Images**
   - Navigate to the `Data_gen` folder and run the following commands:
     - For view distortion: `python3 view.py <input_path> <output_path>`
     - For ghosting + motion distortion: `python3 ghosting_motion.py <input_path> <output_path>`
     - For more ghosting distortion: `python3 ghosting_more.py <input_path> <output_more_ghosting_path>`
     - For motion distortion: `python3 motion.py <input_path> <output_path>`
     - For less ghosting distortion: `python3 ghosting_less.py <input_path> <output_less_ghosting_path>`
     - For missing patch distortion: `python3 missing.py <input_path> <output_path>`

3. **Upload Required Files to Google Drive**
   - Upload the following folders and file to Google Drive:
     - `ACDC_Sample_Data`: Contains ground truth and distorted images.
     - `Unet_traditional`: Contains the traditional approach UNet model and data.
     - `trained_model`: Contains the final trained models (pix2pix, cycleGan, cycleMedGan, proposed).
     - `Msc.ipynb`: Tool for training and inference.

4. **Open and Configure Google Colab**
   - Open `Msc.ipynb` in Google Colab to train and infer the model.

5. **Train the Model**
   - Install dependencies as indicated in the notebook cells.
   - Connect to Google Drive.
   - Specify the path to training source and target:
     - For correction model: `ACDC_Sample_Data/Distorted` (source), `ACDC_Sample_Data/Clean` (target).
     - For proposed model: `ACDC_Sample_Data/Segmented_GT` (target).
   - Set the training mode and model mode:
     - `Train_mode`: Choose from [pix2pix, cyclegan, cyclemedgan, proposed].
     - `Model_mode`: Choose from [pix2pix, cycle_gan, cycle_mgan, proposed].
   - Configure hyperparameters:
     - Batch size: 2 or 4
     - Learning rate: 0.0002
     - Epochs: 1500+ for better results
     - Patch size: 256
   - Specify pretrained network path if continuing training.
   - Run the cells to start training.

6. **Infer the Model**
   - Install dependencies as indicated in the notebook cells.
   - Connect to Google Drive.
   - Navigate to the “Generate prediction(s) from unseen dataset” cell.
   - Specify the source path: `ACDC_Sample_Data/Distorted` and choose a result folder.
   - Set the training mode and model mode:
     - `Train_mode`: Choose from [pix2pix, cyclegan, cyclemedgan, proposed].
     - `Model_mode`: Choose from [pix2pix, cycle_gan, cycle_mgan, proposed].
   - Specify the prediction folder path (e.g., `trained_model/pix2pix`).
   - Set patch size to 256 and checkpoint to latest.
   - Execute the “Inspect the predicted output” cell to view results.

7. **Train and Infer the Traditional UNet Model**
   - Open `Unet.ipynb` in Google Colab.
   - Mount Google Drive and change the directory to `Unet_traditional`.
   - Run cells to start training.
   - For inference, modify `main.py`:
     - Line 391: Change `saved_params_path` to “models/unettrainval.pt”.
     - Set `load_model_params` to true, `save_model_params` to false, `train` to false, and `test` to true.
     - Update the output directory in line 287 under `model_test` function.

8. **Validate Model Output**
   - For correction model: `python3 SSIM.py <input_path_GT> <Input_Output>`
   - For traditional UNet model: `python3 DiceScore.py <input_segmentedout> <input_maskGT>`
   - For proposed network: `python3 proposedvalidation.py <segmentedoutput> <Segmented_GT>`

9. **Check Compactness of Feature Map**
   - Run the following Python code in the `ContrastiveVisual` folder: `python3 TSNE.py`

**Note:** Follow these steps for smooth execution. To reproduce accuracy and results, ensure training on a full ACDC dataset with specified parameters. If issues arise, refresh and reconnect Google Colab for re-execution. Also for further details, please go through report in code repo. 
