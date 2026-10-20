"""Task/model configuration shared by preparation, evaluation, and reporting."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_registry():
    models = json.loads((ROOT / "configs/models.json").read_text())
    tasks = json.loads((ROOT / "configs/tasks.json").read_text())
    if models["seed"] != 1234:
        raise ValueError("Stochastic adapters currently require the frozen seed 1234.")
    return models, tasks


def load_experiments():
    return json.loads((ROOT / "configs/experiments.json").read_text())["experiments"]


def resolve_run(task, model, experiment=None):
    models, tasks = load_registry()
    if task not in tasks["tasks"]:
        raise ValueError(f"Unknown task {task!r}; choose {list(tasks['tasks'])}")
    if model not in models["models"]:
        raise ValueError(f"Unknown model {model!r}; choose {list(models['models'])}")
    model_config, task_config = models["models"][model], tasks["tasks"][task]
    setting = None
    if experiment is not None:
        experiments = load_experiments()
        if experiment not in experiments:
            raise ValueError(f"Unknown experiment {experiment!r}; choose {list(experiments)}")
        setting = experiments[experiment]
        if setting["model"] != model:
            raise ValueError(f"Experiment {experiment!r} requires model {setting['model']!r}")
        model_config = {**model_config, **setting["model_overrides"]}
    shots = task_config["shots_by_family"][model_config["family"]]
    resolved = {
        "model": model_config, "task": task_config, "seed": models["seed"],
        "shots": shots,
        "data_path": ROOT / task_config["data_path"].format(shots=shots),
        "protocol_path": ROOT / task_config["protocol_path"].format(shots=shots),
        "output": ROOT / "results/multitask" / task / model,
    }
    if setting is not None:
        resolved["experiment"] = {"name": experiment, **setting}
        resolved["output"] = ROOT / "results/reduced_depth" / task / setting["output_name"]
    return resolved
