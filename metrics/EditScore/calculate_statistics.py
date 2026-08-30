import argparse
import json
from pathlib import Path

import numpy as np


CATEGORIES = ("prompt_following", "consistency", "overall")
LABELS = {
    "prompt_following": "Prompt Following",
    "consistency": "Consistency",
    "overall": "Overall",
}
TASK_ORDER = (
    "background_change",
    "color_alter",
    "style_change",
    "subject-add",
    "subject-remove",
    "subject-replace",
    "material_alter",
    "motion_change",
    "ps_human",
    "text_change",
    "tone_transfer",
    "extract",
    "compose",
    "average",
)
GROUPS = {
    "object": ("subject-add", "subject-remove", "subject-replace"),
    "appearance": ("color_alter", "material_alter", "style_change", "tone_transfer"),
    "scene": ("background_change", "extract"),
    "advanced": ("ps_human", "text_change", "motion_change", "compose"),
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", type=Path, required=True)
    return parser.parse_args()


def load_scores(path: Path):
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    correct = sum(row["score"][0] > row["score"][1] for row in rows)
    scores = [score for row in rows for score in row["score"]]
    return correct / len(rows), scores


def main(args):
    task_types = sorted(path.name for path in args.result_dir.iterdir())
    results = {category: {} for category in CATEGORIES}
    all_scores = {category: [] for category in CATEGORIES}

    for task_type in task_types:
        task_dir = args.result_dir / task_type
        for category in CATEGORIES:
            accuracy, scores = load_scores(task_dir / f"{category}.jsonl")
            results[category][task_type] = accuracy
            all_scores[category].extend(scores)

    for category in CATEGORIES:
        values = results[category]
        values["average"] = sum(values.values()) / len(values)

    print(" & ".join(TASK_ORDER))
    for category in CATEGORIES:
        values = " & ".join(
            f"{results[category][task]:.3f}" for task in TASK_ORDER
        )
        print(f"{LABELS[category]}: {values}")

    for group_name, group_tasks in GROUPS.items():
        means = [
            np.mean([results[category][task] for task in group_tasks])
            for category in CATEGORIES
        ]
        print(f"{group_name}:")
        print("Prompt Following & Consistency & Overall")
        print(" & ".join(f"{value:.3f}" for value in means))

    print("Average:")
    print("Prompt Following & Consistency & Overall")
    print(
        " & ".join(
            f"{results[category]['average']:.3f}" for category in CATEGORIES
        )
    )

    for category in CATEGORIES:
        scores = all_scores[category]
        stats = (np.min(scores), np.max(scores), np.mean(scores), np.std(scores))
        print(f"{LABELS[category]} Scores:")
        print("Min & Max & Mean & Std")
        print(" & ".join(f"{value:.3f}" for value in stats))


if __name__ == "__main__":
    main(parse_args())
