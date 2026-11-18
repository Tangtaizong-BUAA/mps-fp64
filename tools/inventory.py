"""Record the installed ATen operator universe and native MPS registrations."""

import json
from pathlib import Path
import torch

root = Path(__file__).resolve().parents[1] / "results"
names = sorted(
    n for n in torch._C._dispatch_get_all_op_names() if n.startswith("aten::")
)
registered = [
    n for n in names if torch._C._dispatch_has_kernel_for_dispatch_key(n, "MPS")
]
(root / "mps-registrations.json").write_text(
    json.dumps(
        {
            "torch": torch.__version__,
            "commit": torch.version.git_version,
            "aten_schema_count": len(names),
            "direct_mps_count": len(registered),
            "direct_mps": registered,
            "scope": "Direct MPS dispatch registrations. Includes non-floating and out variants. Composite operations can execute with no direct MPS registration; registered ops can reject specific dtypes or shapes.",
        },
        indent=2,
    )
    + "\n"
)
print(len(names), len(registered))
