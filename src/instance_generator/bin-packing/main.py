import random
from json import dump

from src.consts import INSTANCE_PATH


def generate_bin_packing_instance(n: int, B: int) -> dict:
    """
    Generate a bin packing instance.
    Args:
        n: number of items
        B: bin capacity
        seed: random seed
    Returns:
        dict: bin packing instance
    """
    w = [random.randint(1, B) for _ in range(n)]
    return {
        "num_items": n,
        "bin_capacity": B,
        "item_weights": w,
    }


def main():
    random.seed(42)

    output_dir = INSTANCE_PATH / "bin-packing"
    output_dir.mkdir(parents=True, exist_ok=True)

    for n in range(10, 100, 10):
        for B in range(10, 100, 10):
            instance = generate_bin_packing_instance(n, B)
            output_file = output_dir / f"bin_packing_{n}_{B}.json"
            with open(output_file, "w") as f:
                dump(instance, f, indent=2)


if __name__ == "__main__":
    main()
