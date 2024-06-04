### Prepare Physion Data

Physion v1.5 can be downloaded from https://physion-benchmark.github.io/physion_v15

Unzip the `<type>_movies` file which contains all the scenes.

Build the pkl file for training and validation.

```Usage: python store_physion_filenames.py <directory> <split_ratio> <filename>```

```bash
python physion/store_physion_onthefly_data.py /media/support_data/support_all_movies 0.99 dominoes_may
```

This will store the frames as a single pkl file containing information regarding location of information of each frame in each video.

#### Building physion data (**deprecated**)
> **Deprecated**: This feature is deprecated 
- **Reason for deprecation**: As the new dataloader builds the information (2D --> 3D projection) on the fly, the below code need not be run anymore.

```bash
python physion/store_physion_data.py
```

> **Note**: Make sure to change the ROOT_PATH and other paths in the python script.



