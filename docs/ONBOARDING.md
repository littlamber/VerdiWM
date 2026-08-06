# Universal Model Onboarding

VerdiWM onboards external world-model repositories through a read-only discovery and contract boundary. Common Python projects do not need a handwritten VerdiWM adapter. The generated connector is declarative and lives outside the model source tree.

## Quick Start

```bash
verdiwm-onboard /path/to/model
```

The default sidecar is a sibling directory named `<model>.verdiwm-instance`. An explicit destination is also supported, but it must remain outside the model repository:

```bash
verdiwm-onboard /path/to/model \
  --output-root /path/to/onboarding/model-instance \
  --runtime-python /path/to/environment/bin/python
```

The command performs no dependency installation, model import, training, inference, rollout, or GPU allocation. Dependency imports and `pip check` run in bounded subprocesses with CUDA hidden.

## Sidecar Contract

```text
<model>.verdiwm-instance/
  manifest.json
  model_manifest.json
  runtime_lock.json
  asset_manifest.json
  capability_report.json
  evaluator_contract.json
  generated_connector/
    connector.json
  conformance_report.json
  onboarding-report.json
  onboarding-report.md
```

`manifest.json` is the stable machine entrypoint. `optimization_launch_allowed` remains `false` until a separate conformance runner produces a passing receipt.

## Admission States

| State | Meaning | Scheduler action |
|:--|:--|:--|
| `blocked` | A runtime, source revision, checkpoint, evaluation entrypoint, or evaluator contract is missing or invalid. | Reject. |
| `binding_required` | Discovery completed but a declarative binding is incomplete. | Reject. |
| `ready_for_conformance_smoke` | Runtime and evaluator bindings are complete. Model execution has not been authorized. | Run bounded conformance only. |

The scanner fails closed with stable blocker codes such as `RUNTIME_UNREADY`, `SOURCE_REVISION_UNBOUND`, `CHECKPOINT_MISSING`, `MODEL_ASSET_BINDING_REQUIRED`, `EVALUATION_ENTRYPOINT_MISSING`, and `EVALUATOR_CONTRACT_REQUIRED`. Input-like CLI flags are matched to discovered sidecar assets, so missing model, policy, checkpoint, and dataset paths are rejected before conformance or GPU admission.

## Evaluator Contract

An evaluator contract is an external frozen JSON file. Keeping it outside the imported repository prevents onboarding from rewriting upstream model code or silently changing verdict semantics.

```json
{
  "evaluator_id": "task_success_v1",
  "command": ["python", "scripts/eval.py"],
  "input_artifacts": ["checkpoint", "frozen_split"],
  "output_artifacts": ["evaluation-receipt.json"],
  "metrics": ["success_rate"],
  "verifier": "task_success_receipt_v1"
}
```

Bind it with:

```bash
verdiwm-onboard /path/to/model \
  --runtime-python /path/to/environment/bin/python \
  --evaluator-contract /path/to/contracts/task-success-v1.json
```

Discovery evidence is not a performance claim. Only a passing frozen evaluator receipt may enter reusable optimization memory.
