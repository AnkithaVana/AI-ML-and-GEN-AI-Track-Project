from __future__ import annotations

from src.modeling import train_and_save_model


def main() -> None:
    bundle = train_and_save_model()
    print("Training complete")
    print(f"Best model: {bundle['best_model']}")
    print(f"Data source: {bundle['data_source']}")
    print("Metrics:")
    for row in bundle["metrics"]:
        print(
            f"  {row['model']}: "
            f"accuracy={row['accuracy']}, precision={row['precision']}, "
            f"recall={row['recall']}, f1={row['f1_score']}"
        )


if __name__ == "__main__":
    main()
