# auto_labeling_sam2_refactored.py
import argparse
import cv2
import numpy as np
import torch
from pathlib import Path
from ultralytics import SAM

# --- Constants ---
WINDOW_NAME_LABELING = "SAM Auto-Labeling - Click ALL points, Press 'n' for next, 'q' to quit"
WINDOW_NAME_REVIEW = "Review Labels - 'n': next, 'r': re-label, 'q': quit"


def get_image_files(directory: Path) -> list[Path]:
    """Scans a directory and returns a sorted list of image file paths."""
    image_extensions = ['.png', '.jpg', '.jpeg']
    return sorted([p for p in directory.glob('*') if p.suffix.lower() in image_extensions])


def get_yolo_bbox_from_mask(mask_data: torch.Tensor, img_shape: tuple) -> list[float] | None:
    """
    Converts a single SAM mask into a YOLO format bounding box.

    Args:
        mask_data (torch.Tensor): The mask tensor for a single object from SAM.
        img_shape (tuple): The shape of the original image (height, width).

    Returns:
        list[float] | None: A list containing [x_center, y_center, width, height]
                            in normalized YOLO format, or None if no valid contour is found.
    """
    if mask_data is None or not mask_data.any():
        return None

    # Convert tensor to numpy array for OpenCV processing
    mask_np = mask_data.cpu().numpy().astype(np.uint8)
    contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    # Combine all contours into one to get the overall bounding box
    all_points_contour = np.concatenate(contours)
    x, y, w, h = cv2.boundingRect(all_points_contour)

    img_h, img_w = img_shape[:2]
    x_center = (x + w / 2) / img_w
    y_center = (y + h / 2) / img_h
    width = w / img_w
    height = h / img_h

    return [x_center, y_center, width, height]


class InteractiveSAMLabler:
    """
    A class to handle interactive object labeling using Segment Anything Model (SAM).
    It provides a two-phase workflow: initial labeling and a review session.
    """

    def __init__(self, input_dir: str, output_dir: str, checkpoint: str, class_id: int):
        """
        Initializes the labler with necessary configurations.

        Args:
            input_dir (str): Path to the directory containing images to label.
            output_dir (str): Path to the directory to save YOLO label files.
            checkpoint (str): Path or name of the SAM checkpoint file (e.g., 'sam_l.pt').
            class_id (int): The class ID to be assigned to all labels.
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.checkpoint = checkpoint
        self.class_id = class_id

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")

        self.model = self._load_model()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # State variables for interactive labeling
        self.current_points = []
        self.display_image = None

    def _load_model(self) -> SAM:
        """Loads the SAM model onto the configured device."""
        try:
            print(f"Loading SAM model from checkpoint: {self.checkpoint}")
            model = SAM(self.checkpoint)
            model.to(self.device)
            print(f"SAM model loaded successfully on {self.device}.")
            return model
        except Exception as e:
            print(f"Error loading SAM model: {e}")
            exit()

    def _mouse_callback_labeling(self, event, x, y, flags, param):
        """Mouse callback function to capture points on the image."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.current_points.append([x, y])
            cv2.circle(self.display_image, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow(WINDOW_NAME_LABELING, self.display_image)
            print(f"Clicked point: ({x}, {y})")

    def _process_single_image(self, image_path: Path) -> bool:
        """
        Processes a single image for interactive point-based labeling.

        Args:
            image_path (Path): The path to the image to be labeled.

        Returns:
            bool: True if the process should continue, False if the user quits.
        """
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"Could not read image: {image_path.name}")
            return True  # Continue to the next image

        self.display_image = img.copy()
        self.current_points = []
        label_path = self.output_dir / f"{image_path.stem}.txt"

        print(f"\n--- Labeling image: {image_path.name} ---")
        print("Click ALL points for ALL objects. Press 'n' to save, 'q' to quit.")

        cv2.imshow(WINDOW_NAME_LABELING, self.display_image)

        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('n'):
                break
            elif key == ord('q'):
                print("Quitting labeling session.")
                return False

        if not self.current_points:
            print(f"No points clicked for {image_path.name}. Saving empty label file.")
            label_path.touch() # Create an empty file
            return True

        points_np = np.array(self.current_points)
        results = self.model.predict(img, points=points_np, labels=np.ones(len(points_np)))

        with open(label_path, 'w') as f:
            if results and results[0].masks is not None:
                for mask in results[0].masks.data:
                    bbox = get_yolo_bbox_from_mask(mask, img.shape)
                    if bbox:
                        xc, yc, w, h = bbox
                        f.write(f"{self.class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")
                print(f"Saved {len(results[0].masks.data)} labels for {image_path.name}")
            else:
                print(f"SAM returned no masks for {image_path.name}. Saving empty file.")

        return True

    def _initial_labeling_phase(self):
        """Runs the first phase of labeling for all images without existing labels."""
        print("\n--- Phase 1: Initial Auto-Labeling ---")
        cv2.namedWindow(WINDOW_NAME_LABELING, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(WINDOW_NAME_LABELING, self._mouse_callback_labeling)

        image_files = get_image_files(self.input_dir)
        for image_path in image_files:
            label_path = self.output_dir / f"{image_path.stem}.txt"
            if label_path.exists() and label_path.stat().st_size > 0:
                print(f"Label for {image_path.name} already exists. Skipping.")
                continue

            if not self._process_single_image(image_path):
                break  # User quit the session
        
        cv2.destroyAllWindows()

    def _review_phase(self):
        """Runs the second phase for reviewing and re-labeling existing labels."""
        print("\n--- Phase 2: Review and Re-label ---")
        image_files = get_image_files(self.input_dir)
        
        idx = 0
        while idx < len(image_files):
            image_path = image_files[idx]
            label_path = self.output_dir / f"{image_path.stem}.txt"

            img_review = cv2.imread(str(image_path))
            if img_review is None:
                idx += 1
                continue

            if label_path.exists():
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            _, xc, yc, w, h = map(float, parts)
                            img_h, img_w = img_review.shape[:2]
                            x1 = int((xc - w / 2) * img_w)
                            y1 = int((yc - h / 2) * img_h)
                            x2 = int((xc + w / 2) * img_w)
                            y2 = int((yc + h / 2) * img_h)
                            cv2.rectangle(img_review, (x1, y1), (x2, y2), (0, 255, 0), 2)

            cv2.imshow(WINDOW_NAME_REVIEW, img_review)
            print(f"\nReviewing {image_path.name} ({idx + 1}/{len(image_files)})")
            print("'n': next, 'r': re-label, 'q': quit review")
            
            key = cv2.waitKey(0) & 0xFF

            if key == ord('n'):
                idx += 1
            elif key == ord('r'):
                print(f"Re-labeling {image_path.name}...")
                cv2.destroyWindow(WINDOW_NAME_REVIEW)
                # Re-run labeling for this specific image
                cv2.namedWindow(WINDOW_NAME_LABELING, cv2.WINDOW_NORMAL)
                cv2.setMouseCallback(WINDOW_NAME_LABELING, self._mouse_callback_labeling)
                self._process_single_image(image_path)
                cv2.destroyWindow(WINDOW_NAME_LABELING)
            elif key == ord('q'):
                break
        
        cv2.destroyAllWindows()

    def run(self):
        """Executes the complete labeling and review workflow."""
        self._initial_labeling_phase()
        self._review_phase()
        print("\n--- Process Finished ---")


def main():
    """Main function to parse arguments and run the labeling process."""
    parser = argparse.ArgumentParser(description="SAM Auto-Labeling with Review and Re-labeling")
    parser.add_argument("-i", "--input_dir", type=str, required=True,
                        help="Path to the directory containing images to label.")
    parser.add_argument("-o", "--output_dir", type=str, required=True,
                        help="Path to the directory to save YOLO label files.")
    parser.add_argument("-c", "--checkpoint", type=str, default="sam_l.pt",
                        help="Path or name of the SAM checkpoint. Default: sam_l.pt")
    parser.add_argument("--class_id", type=int, default=0,
                        help="Class ID for the labeled objects. Default: 0")
    args = parser.parse_args()

    labler = InteractiveSAMLabler(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        checkpoint=args.checkpoint,
        class_id=args.class_id
    )
    labler.run()

if __name__ == "__main__":
    main()