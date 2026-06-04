import os, argparse, random

def collect_models(renders_root):
    models = []
    for root, dirs, files in os.walk(renders_root):
        if "images" in dirs:  # only take rendered objects
            rel = os.path.relpath(root, renders_root)
            models.append(rel)
    return models

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--renders-root", required=True,
                    help="Root folder with rendered data (e.g. datasets/renders/modelnet10)")
    ap.add_argument("--out-dir", required=True,
                    help="Where to save split files")
    ap.add_argument("--train", type=float, default=0.8)
    ap.add_argument("--val", type=float, default=0.1)
    ap.add_argument("--test", type=float, default=0.1)
    args = ap.parse_args()

    models = collect_models(args.renders_root)
    random.shuffle(models)

    n = len(models)
    n_train = int(args.train * n)
    n_val = int(args.val * n)

    train_set = models[:n_train]
    val_set = models[n_train:n_train + n_val]
    test_set = models[n_train + n_val:]

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "train.txt"), "w") as f:
        f.write("\n".join(train_set))
    with open(os.path.join(args.out_dir, "val.txt"), "w") as f:
        f.write("\n".join(val_set))
    with open(os.path.join(args.out_dir, "test.txt"), "w") as f:
        f.write("\n".join(test_set))

    print(f"Total models: {n}")
    print(f"Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}")

if __name__ == "__main__":
    main()
