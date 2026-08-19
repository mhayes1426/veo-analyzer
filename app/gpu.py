from __future__ import annotations

import subprocess


def gpu_diagnostics() -> dict:
    result = {"nvidia_runtime": False, "pytorch_installed": False, "cuda_available": False, "gpus": []}
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu,temperature.gpu",
             "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True, timeout=10,
        )
        result["nvidia_runtime"] = True
        for line in completed.stdout.strip().splitlines():
            name, driver, total, used, utilization, temperature = [item.strip() for item in line.split(",")]
            result["gpus"].append({"name": name, "driver": driver, "memory_total_mib": int(total),
                                   "memory_used_mib": int(used), "utilization_percent": int(utilization),
                                   "temperature_c": int(temperature)})
    except (OSError, subprocess.SubprocessError, ValueError):
        pass
    try:
        import torch
        result.update({"pytorch_installed": True, "pytorch_version": torch.__version__,
                       "torch_cuda_version": torch.version.cuda, "cuda_available": torch.cuda.is_available()})
        if torch.cuda.is_available():
            result["cuda_device_count"] = torch.cuda.device_count()
            result["cuda_device_name"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    return result
