# video_frame_extractor.py
import argparse
import cv2
import shutil
import sys
from pathlib import Path
from typing import List

class VideoFrameExtractor:
    """
    A class to extract frames from video files in a specified directory.
    """

    def __init__(self, input_dir: Path, output_dir: Path, frame_interval: int,
                 image_format: str, clean_output: bool):
        """
        Initializes the VideoFrameExtractor.

        Args:
            input_dir (Path): The directory containing video files.
            output_dir (Path): The directory where extracted frames will be saved.
            frame_interval (int): The interval at which to save frames (e.g., 30 means every 30th frame).
            image_format (str): The desired output image format (e.g., '.jpg', '.png').
            clean_output (bool): If True, cleans the output directory before extraction.
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.frame_interval = frame_interval
        self.image_format = self._format_image_extension(image_format)
        self.clean_output = clean_output

        self.total_videos_processed = 0
        self.total_frames_saved = 0

    @staticmethod
    def _format_image_extension(ext: str) -> str:
        """Ensures the image format string starts with a dot."""
        return ext if ext.startswith('.') else f".{ext}"

    def _validate_paths(self):
        """Validates that the input directory exists."""
        if not self.input_dir.is_dir():
            print(f"Error: Input directory not found at '{self.input_dir}'")
            sys.exit(1)

    def _prepare_output_directory(self):
        """Creates the output directory and cleans it if requested."""
        if self.clean_output and self.output_dir.exists():
            print(f"Cleaning output directory: {self.output_dir}")
            try:
                shutil.rmtree(self.output_dir)
            except OSError as e:
                print(f"Error: Could not remove directory {self.output_dir}. Reason: {e}")
                sys.exit(1)

        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            print(f"Output directory is ready at: {self.output_dir}")
        except OSError as e:
            print(f"Error: Could not create directory {self.output_dir}. Reason: {e}")
            sys.exit(1)

    def _scan_for_videos(self) -> List[Path]:
        """Scans the input directory for video files."""
        print(f"\nScanning for video files in: {self.input_dir}")
        # Add other video extensions if needed
        video_extensions = ['.avi', '.mp4', '.mov', '.mkv']
        video_files = [p for p in self.input_dir.glob('*') if p.suffix.lower() in video_extensions]
        
        if not video_files:
            print("No video files found in the specified directory.")
        else:
            print(f"Found {len(video_files)} video file(s).")
        return video_files

    def _process_video(self, video_path: Path):
        """
        Extracts frames from a single video file.

        Args:
            video_path (Path): The path to the video file to process.
        """
        print(f"\nProcessing video: {video_path.name}...")
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            print(f"  Error: Could not open video file.")
            return

        frame_index = 0
        saved_count_this_video = 0
        video_stem = video_path.stem

        while True:
            success, frame = cap.read()
            if not success:
                break # End of video

            if frame_index % self.frame_interval == 0:
                output_filename = f"{video_stem}_frame_{frame_index:06d}{self.image_format}"
                output_path = self.output_dir / output_filename
                
                try:
                    cv2.imwrite(str(output_path), frame)
                    saved_count_this_video += 1
                except Exception as e:
                    print(f"  Error writing frame {output_path}: {e}")

            frame_index += 1

        cap.release()
        self.total_videos_processed += 1
        self.total_frames_saved += saved_count_this_video
        print(f"  Finished processing. Saved {saved_count_this_video} frames.")

    def run(self):
        """Executes the full frame extraction workflow."""
        self._validate_paths()
        self._prepare_output_directory()
        
        video_files = self._scan_for_videos()
        if not video_files:
            return # Exit if no videos are found
        
        for video_path in video_files:
            self._process_video(video_path)

        print("\n--- Extraction Complete ---")
        print(f"Total videos processed: {self.total_videos_processed}")
        print(f"Total frames saved: {self.total_frames_saved}")
        print(f"Output located at: {self.output_dir}")

def main():
    """Parses command-line arguments and runs the frame extractor."""
    parser = argparse.ArgumentParser(
        description="Extract frames from video files in a directory.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("-i", "--input", type=str, required=True,
                        help="Path to the directory containing input video files.")
    parser.add_argument("-o", "--output", type=str, required=True,
                        help="Path to the directory to save the extracted frames.")
    parser.add_argument("-f", "--frame_interval", type=int, default=30,
                        help="Interval to save frames (e.g., 30 means every 30th frame). Default: 30")
    parser.add_argument("--format", type=str, default=".jpg",
                        help="Output image format (e.g., .jpg, .png). Default: .jpg")
    parser.add_argument("--clean", action='store_true',
                        help="If set, the output directory will be cleared before extraction.")

    args = parser.parse_args()
    
    print("--- Frame Extractor Configuration ---")
    print(f"Input Directory: {args.input}")
    print(f"Output Directory: {args.output}")
    print(f"Frame Interval: {args.frame_interval}")
    print(f"Image Format: {args.format}")
    print(f"Clean Output Dir: {'Yes' if args.clean else 'No'}")
    print("-------------------------------------")

    extractor = VideoFrameExtractor(
        input_dir=Path(args.input),
        output_dir=Path(args.output),
        frame_interval=args.frame_interval,
        image_format=args.format,
        clean_output=args.clean
    )
    extractor.run()
    
if __name__ == "__main__":
    main()