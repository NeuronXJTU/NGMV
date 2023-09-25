# Neighbouring-slice Guided Multi-view Framework for Brain Segmentation



This is the official repository for the code related to the article titled "Neighbouring-slice Guided Multi-view Framework for Brain Segmentation" by Xuemeng Hu, Zhongyu Li, Yi Wu, Jingyi Liu, Xiang Luo, and Jing Ren, which was submitted to Nerou Computing in 2023.

## Requirements

Please make sure you have the required packages installed. You can install them using the following command:

```bash
pip install -r requirements.txt
```

## Usage

To run the code, simply execute the `train.py` script. Please note that our training dataset is in TIFF format, and the testing dataset is in NII format. You can adjust the BasicDataset_lsfm/ BasicDataset_res_fvb function according to your dataset or use the provided functions.

### Training

```bash
python train.py
```

### Testing

To evaluate the model, run the `eval_res` function. You can also adjust the data loading or directly call the `BasicDataset_res_test` function to load the testing dataset.

```python
python eval_res.py
```

## Data Preparation

Before running the code, make sure you have prepared your dataset in the following way:

1. Organize the 2D slices of the dataset and shuffle them.
2. Save the slices from different views separately.
3. Adjust the paths to your dataset in the `config.py` file using the following variables:
   - `dataset_path_1`: Path to the first dataset view.
   - `dataset_path_2`: Path to the second dataset view.
   - `label_txt_1`: Path to the text file containing labels for the first view.
   - `label_txt_2`: Path to the text file containing labels for the second view.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Special thanks to the authors of the referenced papers and datasets used in this project.

If you have any questions or encounter any issues, please feel free to contact the authors.