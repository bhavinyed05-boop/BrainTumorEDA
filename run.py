"""Run the complete Brain Tumor MRI exploratory data analysis."""

from pathlib import Path

from src.dashboard import plot_vertical_dashboard
from src.data_pipeline import generate_mri_metadata


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    data_dir = project_dir / "brain_tumor_data" / "Training"
    categories = ["glioma", "meningioma", "notumor", "pituitary"]

    missing_categories = [
        category for category in categories if not (data_dir / category).is_dir()
    ]
    if missing_categories:
        missing = ", ".join(missing_categories)
        raise FileNotFoundError(
            f"Missing dataset category folder(s): {missing}. "
            f"Expected them under {data_dir}"
        )

    metadata = generate_mri_metadata(data_dir, categories)
    if metadata.empty:
        raise RuntimeError(f"No readable MRI images were found under {data_dir}")

    print(f"Loaded {len(metadata)} images.")
    plot_vertical_dashboard(metadata, categories)


if __name__ == "__main__":
    main()
