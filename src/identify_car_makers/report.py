"""Evaluation report management and visualization.

Provides functionality for collecting model predictions, persisting results
to disk, and generating visualizations including confusion matrices and
prediction galleries. Supports CSV serialization with base64-encoded images.
"""

import base64
import csv
import logging
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Self

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from PIL import Image
from sklearn.metrics import confusion_matrix

from .paths import get_path_to_evals

logger = logging.getLogger(__name__)


def _encode_image_to_base64(img: Image.Image) -> str:
    """Convert PIL Image to base64 string for serialization."""
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


def _decode_base64_to_image(encoded_str: str) -> Image.Image:
    """Reconstruct PIL Image from base64 string."""
    img_bytes = base64.b64decode(encoded_str)
    return Image.open(BytesIO(img_bytes))


class EvalReport:
    """Container for evaluation predictions and metrics.
    
    Manages collection of prediction records with ground truth labels,
    supporting persistence to CSV and various visualization methods.
    """

    def __init__(self):
        """Initialize empty report."""
        self.records = []

    def add_record(self, img: Image.Image, truth_label: str, pred_label: str) -> None:
        """Append prediction record to report.

        Args:
            img: Input image for prediction
            truth_label: Ground truth label
            pred_label: Model prediction
        """
        encoded_img = _encode_image_to_base64(img)
        is_match = truth_label == pred_label

        self.records.append({
            "image_base64": encoded_img,
            "ground_truth": truth_label,
            "predicted": pred_label,
            "correct": is_match,
        })

    def to_csv(self) -> str:
        """Persist report to timestamped CSV file.

        Returns:
            Path to created CSV file
        """
        time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"predictions_{time_stamp}.csv"
        output_path = Path(get_path_to_evals()) / filename

        logger.info("Writing evaluation report to: %s", output_path)

        with open(output_path, "w", newline="") as csv_file:
            field_names = ["image_base64", "ground_truth", "predicted", "correct"]
            csv_writer = csv.DictWriter(csv_file, fieldnames=field_names)
            csv_writer.writeheader()
            csv_writer.writerows(self.records)

        logger.info("Report saved with %d records", len(self.records))
        return str(output_path)

    @classmethod
    def from_csv(cls, file_name: str) -> Self:
        """Load report from CSV file in evals directory.

        Args:
            file_name: Name of CSV file to load

        Returns:
            Reconstructed EvalReport instance
        """
        csv_path = Path(get_path_to_evals()) / file_name
        logger.info("Loading evaluation report from: %s", csv_path)

        report_obj = cls()
        csv.field_size_limit(10 * 1024 * 1024)

        with open(csv_path, newline="") as csv_file:
            csv_reader = csv.DictReader(csv_file)
            for entry in csv_reader:
                entry["correct"] = entry["correct"].lower() == "true"
                report_obj.records.append(entry)

        logger.info("Loaded %d records from CSV", len(report_obj.records))
        return report_obj

    @classmethod
    def from_last_csv(cls) -> Self:
        """Load most recent evaluation report from evals directory.

        Returns:
            EvalReport from latest CSV file

        Raises:
            FileNotFoundError: If no CSV files exist in evals directory
        """
        evals_dir = Path(get_path_to_evals())
        csv_files = sorted(
            evals_dir.glob("predictions_*.csv"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        if not csv_files:
            raise FileNotFoundError(f"No prediction CSV files found in {evals_dir}")

        most_recent = csv_files[0]
        logger.info("Loading most recent report: %s", most_recent.name)
        return cls.from_csv(most_recent.name)

    def print(self, only_misclassified: bool = False):
        """Generate visualization grid of predictions.

        Args:
            only_misclassified: If True, show only incorrect predictions

        Returns:
            Matplotlib figure containing image grid
        """
        display_records = (
            [rec for rec in self.records if not rec["correct"]]
            if only_misclassified
            else self.records
        )

        total_images = len(display_records)
        grid_cols = 4
        grid_rows = (total_images + grid_cols - 1) // grid_cols

        fig, axes_array = plt.subplots(grid_rows, grid_cols, figsize=(15, grid_rows * 4))
        axes_flat = axes_array.flatten() if total_images >= 1 else [axes_array]

        for idx, record in enumerate(display_records):
            img = _decode_base64_to_image(record["image_base64"])
            axes_flat[idx].imshow(img)
            axes_flat[idx].axis("off")

            title_color = "green" if record["correct"] else "red"
            title_text = f"GT: {record['ground_truth']}\nPred: {record['predicted']}"
            axes_flat[idx].set_title(title_text, color=title_color, fontweight="bold")

        for idx in range(total_images, len(axes_flat)):
            axes_flat[idx].axis("off")

        plt.tight_layout()
        plt.close(fig)
        return fig

    def get_accuracy(self) -> float:
        """Calculate overall prediction accuracy.

        Returns:
            Accuracy as fraction of correct predictions
        """
        if not self.records:
            return 0.0
        
        num_correct = sum(1 for rec in self.records if rec["correct"])
        return num_correct / len(self.records)

    def print_confusion_matrix(self) -> None:
        """Display confusion matrix heatmap for predictions.
        
        Visualizes classification performance across all label classes
        using a confusion matrix with counts and accuracy statistics.
        """
        truth_labels = [rec["ground_truth"] for rec in self.records]
        pred_labels = [rec["predicted"] for rec in self.records]

        unique_classes = sorted(set(truth_labels + pred_labels))
        conf_matrix = confusion_matrix(truth_labels, pred_labels, labels=unique_classes)

        plt.figure(figsize=(12, 10))
        sns.heatmap(
            conf_matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=unique_classes,
            yticklabels=unique_classes,
            cbar_kws={"label": "Number of Predictions"},
        )

        plt.title(
            "Confusion Matrix: Predicted vs Actual Car Makers",
            fontsize=16,
            fontweight="bold",
        )
        plt.xlabel("Predicted Class", fontsize=12)
        plt.ylabel("Actual Class", fontsize=12)
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()

        diagonal_sum = np.trace(conf_matrix)
        total_preds = len(truth_labels)
        accuracy = diagonal_sum / total_preds

        logger.info("Number of classes: %d", len(unique_classes))
        logger.info("Classes: %s", unique_classes)
        logger.info("Total predictions: %d", total_preds)
        logger.info("Correct predictions: %d", diagonal_sum)
        logger.info("Accuracy: %.3f", accuracy)


if __name__ == "__main__":
    report = EvalReport.from_last_csv()
    logger.info("Loaded %d records from latest CSV", len(report.records))
    
    report.print(only_misclassified=True)
    report.print_confusion_matrix()
