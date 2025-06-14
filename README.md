# Python-Powered Dataset Toolkit for Computer Vision

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

A collection of powerful and easy-to-use Python scripts designed to streamline the workflow for preparing computer vision datasets, especially for YOLOv8.

---

### About Me

Hi there! I'm **Duc Thang**, a passionate developer with a keen interest in Artificial Intelligence and Computer Vision. I created this toolkit to solve the common, repetitive, and often tedious tasks I encountered while working on object detection projects. My goal is to automate the data preparation pipeline, allowing developers and researchers to focus more on model training and experimentation.

### Project Idea

The core idea behind this project is to provide a comprehensive set of command-line tools that cover the entire dataset preparation lifecycle:

1.  **Frame Extraction:** Automatically extract frames from video files to build an initial image pool.
2.  **Interactive Labeling:** Use the power of the Segment Anything Model (SAM) for rapid, semi-automated object annotation.
3.  **Dataset Splitting:** Effortlessly split the labeled data into `train` and `val` sets, conforming to the YOLOv8 directory structure, and automatically generate the necessary `data.yaml` file.

This toolkit aims to make the journey from raw videos to a train-ready dataset as smooth and efficient as possible.

---

## 1. Requirements

Before you begin, ensure you have the following set up:

### Hardware

*   **CPU:** A modern multi-core processor is sufficient for most tasks.
*   **GPU:** A CUDA-enabled NVIDIA GPU is **highly recommended** for the interactive labeling script (`auto_labeling_sam2_refactored.py`), as it significantly speeds up the Segment Anything Model (SAM). The other scripts run efficiently on a CPU.
*   **RAM:** 16GB or more is recommended, especially when handling high-resolution images.

### Software

*   **Python:** Version 3.10 or newer.
*   **Git:** To clone this repository.
*   **Pip:** For installing Python packages.

## 2. Installation

## 2. Installation

You can set up the project using one of the following methods.

### Method 1: Using the installation script (Recommended for Linux/macOS)

This automated script will create a virtual environment, detect your hardware (CPU/GPU), and install the appropriate dependencies.

1.  **Clone the repository:**
    ```bash
    https://github.com/ThangDuc3101/auto_labeling_with_SAM2.git
    cd auto_labeling_with_SAM2
    ```

2.  **Run the installation script:**
    ```bash
    chmod +x install.sh
    ./install.sh
    ```
    After the script finishes, the environment will be ready to use. To activate it later, run `source venv/bin/activate`.

### Method 2: Manual installation using pip (For all systems)

This method gives you more control and is suitable for Windows users.

1.  **Clone the repository and navigate into it:**
    ```bash
    git clone https://github.com/YourUsername/your-repository-name.git
    cd your-repository-name
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    # For Windows
    python -m venv venv
    .\venv\Scripts\activate

    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install dependencies from `requirements.txt`:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **(GPU Users Only) Install PyTorch with CUDA support:**
    The command above installs the CPU version of PyTorch. If you have a CUDA-enabled GPU, you must perform this extra step for better performance:
    *   First, uninstall the CPU versions: `pip uninstall torch torchvision torchaudio`
    *   Then, visit the [official PyTorch website](https://pytorch.org/get-started/locally/) to find the correct installation command for your specific CUDA version and run it.
   
5.  **Download a SAM Checkpoint:**
    The auto-labeling script requires a pre-trained SAM checkpoint. Download one and place it in the project's root directory. The default script looks for `sam_l.pt`.
    *   [sam_l.pt (Large model)](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth) -> Rename to `sam_l.pt`
    *   [sam_b.pt (Base model - smaller & faster)](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth) -> Rename to `sam_b.pt`

## 3. Usage Workflow

The scripts are designed to be used in a sequential workflow. Here is the recommended order of operations:

### Step 1: Extract Frames from Videos

Use `video_frame_extractor.py` to convert your raw videos into a collection of images.

*   **Purpose:** Create a static dataset of images from video footage.
*   **Command:**

    ```bash
    python video_frame_extractor.py -i path/to/your/videos -o path/to/save/images -f 30 --clean
    ```

    *   `-i`: Input directory containing your `.mp4`, `.avi`, etc. files.
    *   `-o`: Output directory where the extracted frames will be saved.
    *   `-f`: Frame interval. `30` means one frame will be saved every 30 frames. Adjust as needed.
    *   `--clean`: (Optional) Deletes the output directory before starting, ensuring a fresh extraction.

### Step 2: Label Your Images

Use `auto_labeling_sam2_refactored.py` for a fast, interactive labeling experience. This script has two phases: initial labeling and a review session.

*   **Purpose:** Create YOLO-format `.txt` label files for your images.
*   **Command:**

    ```bash
    python auto_labeling_sam2_refactored.py -i path/to/your/images -o path/to/save/labels --class_id 0
    ```

    *   `-i`: The directory containing the images you extracted in Step 1.
    *   `-o`: The directory where the `.txt` label files will be saved.
    *   `--class_id`: The integer ID for the class you are labeling (e.g., `0` for 'person').
    *   You can also specify a different SAM checkpoint with `-c sam_b.pt`.

### Step 3: Split the Dataset and Generate YAML

Finally, use `split_dataset.py` to organize your labeled data into `train` and `val` sets and create the `data.yaml` file required for training.

*   **Purpose:** Prepare the final YOLOv8-compatible dataset structure.
*   **Command:**

    ```bash
    python split_dataset.py -i path/to/your/images -l path/to/your/labels -o path/to/final_dataset -r 0.8 --nc 1 -n person
    ```

    *   `-i`: Your image directory.
    *   `-l`: Your label directory.
    *   `-o`: The final output directory where the complete, structured dataset will be created.
    *   `-r`: The training/validation split ratio (`0.8` means 80% for training, 20% for validation).
    *   `--nc`: The total number of classes.
    *   `-n`: A space-separated list of class names (e.g., `-n person car bicycle`).

After completing these three steps, the `final_dataset` directory will be ready to be used directly for training a YOLOv8 model.

*--- GOOD LUCK ---*