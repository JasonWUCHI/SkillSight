import argparse
import json
import os

import numpy as np
from fvcore.common.file_io import PathManager


def balanced_accuracy_from_confmat(sc_total):
    per_class_recall = np.diag(sc_total) / sc_total.sum(axis=1)
    per_class_recall = np.nan_to_num(per_class_recall)
    return per_class_recall.mean()


def main():
    parser = argparse.ArgumentParser(description="Evaluate saved segment prediction JSONs at video level.")
    parser.add_argument("--split-csv", required=True)
    parser.add_argument("--json-dir", required=True)
    parser.add_argument("--scenarios", default="Basketball,Soccer,RockClimbing,Music,Dance,Cooking")
    args = parser.parse_args()

    label_dict = {}
    with PathManager.open(args.split_csv, "r") as f:
        for path_label in f.read().splitlines():
            path, label = path_label.split(" ")
            label_dict[path] = int(label)

    total = [[0, 0, 0, 0] for _ in range(4)]
    for sc in [s for s in args.scenarios.split(",") if s]:
        sc_total = [[0, 0, 0, 0] for _ in range(4)]
        with open(os.path.join(args.json_dir, f"{sc}.json"), "r") as f:
            data = json.load(f)

        for key in data:
            path = data[key][0]["path"]
            path = path[path.index("takes"):]
            preds = np.zeros((4))
            for prediction in data[key]:
                preds += np.array(prediction["pred"])
            preds /= len(data[key])
            pred_class = int(np.argmax(preds))
            label = label_dict[path]
            total[label][pred_class] += 1
            sc_total[label][pred_class] += 1

        print(f"Scenario: {sc}, balanced_acc={balanced_accuracy_from_confmat(np.array(sc_total)):.4f}")

    print(f"Overall balanced_acc={balanced_accuracy_from_confmat(np.array(total)):.4f}")
    for row in total:
        print(row)


if __name__ == "__main__":
    main()
