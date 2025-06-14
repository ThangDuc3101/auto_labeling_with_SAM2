# split_dataset.py
import argparse
import random
import shutil
import sys
import yaml
from pathlib import Path
from typing import List, Optional

class DatasetSplitter:
    """
    A class to split an image dataset into training and validation sets
    for YOLOv8, and generate the corresponding `data.yaml` file.
    """

    def __init__(self, image_dir: Path, label_dir: Path, output_dir: Path,
                 train_ratio: float, num_classes: Optional[int], class_names: Optional[List[str]]):
        """
        Initializes the DatasetSplitter.

        Args:
            image_dir (Path): Directory containing the source images.
            label_dir (Path): Directory containing the source labels.
            output_dir (Path): Directory where the split dataset will be saved.
            train_ratio (float): The proportion of the dataset to allocate for training (e.g., 0.8).
            num_classes (Optional[int]): The total number of classes in the dataset.
            class_names (Optional[List[str]]): A list of the names for each class.
        """
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.output_dir = output_dir
        self.train_ratio = train_ratio
        self.num_classes = num_classes
        self.class_names = class_names

        # Define output paths
        self.train_images_path = self.output_dir / 'images' / 'train'
        self.val_images_path = self.output_dir / 'images' / 'val'
        self.train_labels_path = self.output_dir / 'labels' / 'train'
        self.val_labels_path = self.output_dir / 'labels' / 'val'

    def _validate_inputs(self):
        """Validates the input directories and parameters."""
        if not self.image_dir.is_dir() or not self.label_dir.is_dir():
            print(f"Error: Source image or label directory does not exist.")
            sys.exit(1)

        if not 0 < self.train_ratio < 1:
            print(f"Error: Train ratio must be between 0 and 1.")
            sys.exit(1)

        if self.num_classes is not None and self.class_names is not None:
            if self.num_classes != len(self.class_names):
                print(f"Error: Number of classes ({self.num_classes}) does not match "
                      f"the number of class names provided ({len(self.class_names)}).")
                sys.exit(1)

    def _collect_and_validate_files(self) -> List[Path]:
        """
        Collects all valid image-label pairs from the source directories.

        Returns:
            List[Path]: A list of Path objects for the images that have a corresponding label.
        """
        print("Collecting and validating files...")
        image_extensions = ['.jpg', '.jpeg', '.png']
        all_images = [p for p in self.image_dir.iterdir() if p.suffix.lower() in image_extensions]
        
        valid_image_paths = []
        for img_path in all_images:
            label_path = self.label_dir / f"{img_path.stem}.txt"
            if label_path.exists():
                valid_image_paths.append(img_path)
            else:
                print(f"Warning: Skipping '{img_path.name}' as no corresponding label file was found.")
        
        if not valid_image_paths:
            print("Error: No valid image-label pairs found. Aborting.")
            sys.exit(1)
            
        print(f"Found {len(valid_image_paths)} valid image-label pairs.")
        return valid_image_paths

    def _create_output_directories(self):
        """Creates the required directory structure for the output dataset."""
        print("\nCreating output directory structure...")
        for path in [self.train_images_path, self.val_images_path, self.train_labels_path, self.val_labels_path]:
            path.mkdir(parents=True, exist_ok=True)

    def _copy_files(self, file_paths: List[Path], image_dest: Path, label_dest: Path):
        """
        Copies image and label files to their destination directories.

        Args:
            file_paths (List[Path]): List of image paths to copy.
            image_dest (Path): Destination directory for images.
            label_dest (Path): Destination directory for labels.
        """
        for img_path in file_paths:
            label_path = self.label_dir / f"{img_path.stem}.txt"
            shutil.copy(img_path, image_dest)
            shutil.copy(label_path, label_dest)

    def _create_yaml_file(self):
        """Creates the data.yaml file for YOLO training."""
        print("\nCreating data.yaml file...")
        
        yaml_data = {
            'path': str(self.output_dir.resolve()),
            'train': 'images/train',
            'val': 'images/val',
        }
        
        if self.num_classes is not None:
            yaml_data['nc'] = self.num_classes
        else:
            yaml_data['nc'] = '??? # TODO: Please update the number of classes!'
        
        if self.class_names is not None:
            yaml_data['names'] = self.class_names
        else:
            yaml_data['names'] = ['name1', 'name2', '...'] # TODO: Please update the class names!

        yaml_file_path = self.output_dir / 'data.yaml'
        with open(yaml_file_path, 'w', encoding='utf-8') as f:
            yaml.dump(yaml_data, f, sort_keys=False, allow_unicode=True)
            
        print(f"Successfully created '{yaml_file_path}'")

    def split(self):
        """Executes the entire dataset splitting process."""
        self._validate_inputs()
        
        valid_files = self._collect_and_validate_files()
        
        random.shuffle(valid_files)
        
        split_index = int(len(valid_files) * self.train_ratio)
        train_files = valid_files[:split_index]
        val_files = valid_files[split_index:]

        print(f"\nTotal files: {len(valid_files)}")
        print(f"Training set: {len(train_files)} files")
        print(f"Validation set: {len(val_files)} files")

        self._create_output_directories()

        print("\nCopying training files...")
        self._copy_files(train_files, self.train_images_path, self.train_labels_path)
        
        print("Copying validation files...")
        self._copy_files(val_files, self.val_images_path, self.val_labels_path)
        
        self._create_yaml_file()

        print("\nProcess completed successfully!")
        print(f"Split dataset is located in: {self.output_dir}")
        
        if self.num_classes is None or self.class_names is None:
            print("\n!!! IMPORTANT !!!")
            print(f"Please manually update 'nc' and 'names' in '{self.output_dir / 'data.yaml'}' before training.")
        else:
            print("\nDataset is ready for training!")


def main():
    """Parses command-line arguments and initiates the dataset splitting process."""
    parser = argparse.ArgumentParser(
        description="Split a dataset for YOLOv8 and create the data.yaml file.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('-i', '--images', type=str, required=True,
                        help="Path to the source images directory.")
    parser.add_argument('-l', '--labels', type=str, required=True,
                        help="Path to the source labels directory.")
    parser.add_argument('-o', '--output', type=str, required=True,
                        help="Path to the output directory for the split dataset.")
    parser.add_argument('-r', '--ratio', type=float, default=0.8,
                        help="Ratio for the training set (e.g., 0.8 for 80%). Default: 0.8")
    parser.add_argument('--nc', type=int,
                        help="The number of classes in the dataset.")
    parser.add_argument('-n', '--names', nargs='+', type=str,
                        help="A list of class names, separated by spaces.\nExample: -n cat dog person")
    
    args = parser.parse_args()

    splitter = DatasetSplitter(
        image_dir=Path(args.images),
        label_dir=Path(args.labels),
        output_dir=Path(args.output),
        train_ratio=args.ratio,
        num_classes=args.nc,
        class_names=args.names
    )
    splitter.split()

if __name__ == '__main__':
    main()