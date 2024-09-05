# Eagle-X5 Caption Batch
This tool uses the [Eagle-X5-7B](https://huggingface.co/NVEagle/Eagle-X5-7B) model from NVIDIA to generate keyword-based captions for images in an input folder. Special thanks to NVIDIA for training this powerful model.

It's a fast and robust captioning model that produces comma-separated keyword outputs.

## Requirements
* Python 3.10 or above.
  * It's been tested with 3.10, 3.11 and 3.12.
  * It does not work with 3.8.

* Cuda 12.1.
  * It may work with other versions. Untested.
 
To use CUDA / GPU speed captioning, you'll need ~6GB VRAM or more.

## Setup
1. Create a virtual environment. Use the included `venv_create.bat` to automatically create it. Use python 3.10 or above.
2. Install the libraries in requirements.txt. `pip install -r requirements.txt`. This is done by step 1 when asked if you use `venv_create`.
3. Install [Pytorch for your version of CUDA](https://pytorch.org/). It's only been tested with version 12.1 but may work with others.
4. Open `batch.py` in a text editor and change the BATCH_SIZE = 7 value to match the level of your GPU.

>   For a 6gb VRAM GPU, use 1.
>   For a 24gb VRAM GPU, use 7.

## How to use
1. Activate the virtual environment. If you installed with `venv_create.bat`, you can run `venv_activate.bat`.
2. Run `python batch.py` from the virtual environment.

This runs captioning on all images in the /input/-folder.

## Credits
Thanks to [MNeMoNiCuZ](https://github.com/MNeMoNiCuZ) for the original script upon which this one is based, and [Gökay Aydoğan](https://huggingface.co/gokaygokay) for additional script support.