```markdown
# GPU Occupier

GPU Occupier is a Python package designed to monitor and **occupy** GPU resources efficiently. This package is particularly useful for users who need to manage GPU memory usage for various applications, such as deep learning and data processing.

If you encounter any issues while using this project, feel free to contribute and help improve it.

## Features

- Monitor GPU memory usage in real-time.
- Automatically **occupy** GPU memory based on configurable thresholds.
- Execute commands on specified GPUs while managing memory occupation.
- Support for multiple GPUs.

## Installation

You can install the GPU occupier package using pip. First, clone the repository:

```bash
git clone https://github.com/shu1l0n9/gpu-occupier.git
cd gpu-occupier
```

Then, install the package:

```bash
pip install .
```

```bash
gpu-occupier
```

## Usage Examples

### Example 1: Basic Usage - Monitor and **Occupy** a Single GPU

```bash
gpu-occupier --threshold 20 --ratio 0.8 --timeout 2
```

This will:

- Check GPU memory usage every 5 seconds
- **Occupy** an available GPU when idle
- **Occupy** 80% of memory when usage is below 20%
- Release the GPU and exit after 2 minutes of occupation (Recommanded)

### Example 2: Automatically Select and **Occupy** Multiple GPUs

```bash
python gpu_occupier.py --num-gpus 2 --threshold 20 --ratio 0.7 --timeout 12
```

This will:

- Automatically select 2 GPUs with the lowest memory usage (below 20%)
- **Occupy** 70% of memory on each GPU
- Exit automatically after 12 minutes if u do nothing

### Example 3: Execute a Command After Occupation

```bash
python gpu_occupier.py --gpus 0,1 --threshold 10 --command "python train.py" --work-dir "/path/to/project" --exit-after-command
```

This will:

- Monitor GPUs 0 and 1
- **Occupy** the default ratio (70%) of memory when usage is below 10%
- Execute the training command in the specified working directory
- Exit automatically after the command completes

### Example 4: Use English Interface

```bash
python gpu_occupier.py --gpus 0 --threshold 25 --ratio 0.5 --lang en
```

This will:

- Monitor GPU 0
- Use a 25% threshold and **occupy** 50% of memory
- Display all messages in English

## Notes

1. The program requires access to NVIDIA drivers and GPU devices.
2. The memory occupation ratio should not be set too high; reserve some memory for system use.
3. When using `--exit-after-command`, the program will exit immediately after the command completes.
4. The program can be terminated early using Ctrl+C or the `kill` command.
5. Upon termination, the program will automatically release occupied memory and terminate child processes.

## Parameter Description

| Parameter        | Short | Type   | Default | Description                                                          |
|-----------------|-------|--------|---------|----------------------------------------------------------------------|
| `--gpus`        | None  | str    | None    | Comma-separated list of GPU IDs to monitor (e.g., '0,1,3')        |
| `--total-gpus`  | None  | int    | None    | Total number of GPUs in the system (deprecated)                    |
| `--num-gpus`    | None  | int    | 1       | Number of GPUs to **use**. The program will try to select the GPUs with the least memory usage.                                             |
| `--threshold`   | None  | float  | 20      | Memory usage threshold (%) to trigger occupation                   |
| `--interval`    | None  | float  | 5       | Check interval (seconds)                                            |
| `--ratio`       | None  | float  | 0.7     | Memory ratio to occupy when triggered (0-1)                       |
| `--timeout`     | None  | float  | 2       | Maximum runtime (minutes)                                           |
| `--command`     | None  | str    | None    | Command to execute                                                   |
| `--work-dir`    | None  | str    | None    | Working directory for command execution                            |
| `--exit-after-command` | None  | bool   | False   | Whether to exit after command execution                            |
| `--lang`        | None  | str    | 'zh'    | Output message language (en/zh)                                    |

## Advanced Usage

### Run in Background with nohup

```bash
nohup python gpu_occupier.py --gpus 0 --threshold 20 --timeout 240 > gpu_occupier.log 2>&1 &
```

### Monitor Multiple GPUs and Execute Different Commands

You can write a wrapper script to launch separate processes for each GPU:

```bash
#!/bin/bash
# Launch occupation and execute Command A on GPU 0
python gpu_occupier.py --gpus 0 --command "python task_a.py" --work-dir "/path/to/task_a" &

# Launch occupation and execute Command B on GPU 1
python gpu_occupier.py --gpus 1 --command "python task_b.py" --work-dir "/path/to/task_b" &

wait
```

## FAQ

**Q: What if the program fails to detect my GPUs or if it's unable to access them?**

A: If the program can't find your GPUs, or if it's unable to access them, please check the following:

1.  **NVIDIA Drivers and CUDA:** Make sure you have properly installed the NVIDIA drivers and, optionally, the CUDA toolkit.
2.  **`nvidia-smi` Command:** Verify that the `nvidia-smi` command is working correctly in your terminal.  This is the primary tool the program uses to gather GPU information.
3.  **PyTorch (if applicable):** If you're using this program in conjunction with PyTorch, confirm that PyTorch is installed with CUDA support (e.g., using the appropriate `pip install` command with CUDA dependencies).

**Q: Why doesn't GPU utilization (compute usage) change after memory is occupied?**

A: That's expected behavior. This program primarily **occupies** GPU *memory*.  It doesn't directly manipulate the GPU's compute units (i.e., utilization/compute usage). You'll see changes in the "Memory" column when running `nvidia-smi`.

**Q: How can I verify the program is running?**

A: You can confirm that the program is active in a couple of ways:

1.  **`nvidia-smi`:**  Use the `nvidia-smi` command in your terminal. You should see an increase in memory usage (under the "Memory-Usage" or similar heading) on the GPUs specified in your command-line arguments.
2.  **Process List:** You can also check the process list (e.g., using `ps aux | grep gpu_occupier.py`) to see if the Python process is running.
3. **System Monitoring Tools:**  You can use system monitoring tools (like `top`, `htop`, or `Task Manager` on some systems) to check for the Python process and its resource usage.

**Q: What if memory is not released after an unexpected termination of the program?**

A: The program is designed to release occupied GPU memory when it's terminated gracefully (e.g., by pressing Ctrl+C).  However, if the program crashes or is killed abruptly, memory might not be released immediately.  In this situation:

1.  **Manual Check:**  Run the `nvidia-smi` command.
2.  **Identify and Terminate:**  If you see processes occupying GPU memory that should have been released, manually terminate those processes using the `kill` command (e.g., `kill <process_id>`). You can find the process ID from `nvidia-smi`.
If you frequently encounter this issue, you might want to investigate potential causes of the crashes, such as out-of-memory errors in the command being run.  Increasing the memory usage threshold or decreasing the occupation ratio could help mitigate this.
```
