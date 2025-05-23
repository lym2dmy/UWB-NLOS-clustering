# UWB LOS, Hard NLOS, and Soft NLOS Classification Dataset

This repository contains a dataset for classifying Ultra-Wideband (UWB) signal propagation environments into three categories:

- **LOS (Line-of-Sight)**
- **Hard NLOS (Non-Line-of-Sight with severe obstructions)**
- **Soft NLOS (Non-Line-of-Sight with mild or partial obstructions)**

The dataset is stored in a CSV file named `3class.csv` and is intended for machine learning and signal processing research focused on indoor localization and wireless channel characterization.

## Dataset Overview

The file `3class.csv` contains labeled data samples with features extracted from UWB signal measurements. Each row in the CSV corresponds to a single UWB measurement instance.

### Features

- Various statistical and physical attributes derived from the UWB Channel Impulse Response (CIR), such as amplitude, delay, energy, etc.  
- Feature names and values vary depending on the UWB signal processing method used.

### Label

- The final column indicates the class label:
  - `0` — LOS
  - `1` — Hard NLOS
  - `2` — Soft NLOS

## Usage

You can load the dataset using common Python libraries such as pandas:

```python
import pandas as pd

# Load the dataset
df = pd.read_csv('3class.csv')

# Display basic info
print(df.head())
```

This dataset is suitable for training and evaluating classification models using tools like scikit-learn, TensorFlow, or PyTorch.

## Applications

- UWB-based indoor positioning systems
- Wireless signal propagation analysis
- AI-driven environment classification

## License

This dataset is provided for research and educational purposes. Please contact the maintainer for commercial usage inquiries.

## Citation

If you use this dataset in your research, please cite it appropriately: Barral, V., Escudero, C.J., García-Naya, J.A. and Maneiro-Catoira, R., 2019. NLOS identification and mitigation using low-cost UWB devices. Sensors, 19(16), p.3464.

