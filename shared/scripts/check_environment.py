import importlib.util, json, platform, subprocess, sys
from pathlib import Path

def command(args):
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None

report = {"python": sys.version.split()[0], "platform": platform.platform(), "architecture": platform.machine(), "memory_bytes": command(["sysctl", "-n", "hw.memsize"]), "packages": {name: bool(importlib.util.find_spec(name)) for name in ["torch", "transformers", "mlx", "fastapi", "numpy"]}, "mps_test": "not_tested"}
if report["packages"]["torch"]:
    try:
        import torch
        report["torch_version"] = torch.__version__
        report["mps_available"] = torch.backends.mps.is_available()
        if report["mps_available"]:
            a = torch.randn(16, 16, device="mps", requires_grad=True)
            (a @ a.T).sum().backward()
            torch.mps.synchronize()
            report["mps_test"] = "small_forward_backward_passed"
    except Exception as exc:
        report["mps_test"] = "failed"
        report["error"] = str(exc)
out = Path(__file__).resolve().parents[2] / "environment-check.json"
out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(report, ensure_ascii=False, indent=2))
