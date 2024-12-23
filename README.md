# Data Centric Track

This is the official GitHub of the **Data Centric Track** of the Wake Vision Challenge (TBD: link to the challenge).

It asks participants to **push the boundaries of tiny computer vision** by enhancing the data quality of the newly released [Wake Vision](https://wakevision.ai/), a person detection dataset.

Participants will be able to **modify** as they want the **provided data**. 

The quality increments will be evaluated by training the [mcunet-vww2 model](https://github.com/mit-han-lab/mcunet), a state-of-the-art model in person detection, on the resulting dataset.

## To Get Started

Create a new environment.

```
python -m venv /path/to/new/virtual/environment
```

Activate the environment.

```
source /path/to/new/virtual/environment/bin/activate
```

Install requirements.

```
python -m pip install -r requirements.txt
```

In "data_centric_track.py" modify the value of the variable "data_dir", writing the path to the location in which you would like to save the dataset (239.25 GiB).

The first execution will require hours, since it has to download the entire dataset. It will train the [mcunet-vww2 model](https://github.com/mit-han-lab/mcunet) on the original dataset to get you started. Then you can modify the script as you like, and propose your own modifications to the dataset (just don't change the model architecture).

```
python data_centric_track.py
```

