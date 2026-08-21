from fair_hpo.config.loader import (
    load_dataset_config,
    load_fairness_config,
)


def main():
    datasets = load_dataset_config()
    fairness = load_fairness_config()

    print("===== DATASET CONFIG =====")

    for name, config in datasets["datasets"].items():
        print(
            f"{name}: "
            f"target={config['target_column']}"
        )

    print()
    print("===== FAIRNESS CONFIG =====")

    for name, config in fairness["datasets"].items():
        print(
            f"{name}: "
            f"enabled={config['enabled']}, "
            f"sensitive={config['sensitive_attribute']}"
        )

    print()
    print("CONFIG TEST: PASS")


if __name__ == "__main__":
    main()