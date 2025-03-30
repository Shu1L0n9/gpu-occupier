#!/usr/bin/env python3

import os
import time
import signal
import subprocess
import torch
import argparse
import threading
import sys
import datetime

# JSON content for different log messages (EN and ZH)
json_messages = {
    "get_gpu_info_fail": {
        "en": "❌ Failed to get GPU information: {error}",
        "zh": "❌ 获取GPU信息失败: {error}"
    },
    "monitoring_gpu": {
        "en": "🔍 Monitoring GPU {gpu_id}",
        "zh": "🔍 监控GPU {gpu_id}"
    },
    "gpu_config": {
        "en": "⚙️ GPU {gpu_id} Configuration: Threshold={threshold}%, Check Interval={check_interval}s, Occupation Ratio={occupation_percentage}%",
        "zh": "⚙️ GPU {gpu_id} 配置: 阈值={threshold}%, 检查间隔={check_interval}秒, 占用比例={occupation_percentage}%"
    },
    "max_runtime": {
        "en": "⏱️ Maximum Runtime: {max_runtime_minutes} minutes",
        "zh": "⏱️ 最大运行时间: {max_runtime_minutes} 分钟"
    },
    "will_execute_command": {
        "en": "🖥️ Will execute command: {command}",
        "zh": "🖥️ 将执行命令: {command}"
    },
    "working_directory": {
        "en": "📁 Working Directory: {work_dir}",
        "zh": "📁 工作目录: {work_dir}"
    },
    "continue_after_command": {
        "en": "🔄 Will continue to occupy after command execution",
        "zh": "🔄 命令执行完毕后继续占用"
    },
    "exit_after_command": {
        "en": "🔄 Will exit after command execution",
        "zh": "🔄 命令执行完毕后将退出"
    },
    "occupy_memory_start": {
        "en": "🚀 GPU {gpu_id} memory usage is below the threshold, starting to occupy...",
        "zh": "🚀 GPU {gpu_id} 显存使用率低于阈值，开始占用..."
    },
    "occupy_memory_success": {
        "en": "✅ Successfully occupied GPU {gpu_id}! Current memory usage: {usage:.2f}%",
        "zh": "✅ 成功占用GPU {gpu_id}! 当前显存使用率: {usage:.2f}%"
    },
    "occupy_memory_fail": {
        "en": "❌ Failed to occupy memory on GPU {gpu_id}: {error}",
        "zh": "❌ GPU {gpu_id} 占用显存失败: {error}"
    },
    "execute_command_start": {
        "en": "🔄 Executing command on GPU {gpu_id}: {command}",
        "zh": "🔄 正在GPU {gpu_id} 上执行命令: {command}"
    },
    "command_started": {
        "en": "✅ Command started on GPU {gpu_id}, PID: {pid}",
        "zh": "✅ 命令已启动在GPU {gpu_id} 上，PID: {pid}"
    },
    "command_exited": {
        "en": "⏳ Command exited, exiting the GPU occupation program immediately",
        "zh": "⏳ 命令已退出，立即退出GPU占用程序"
    },
    "command_start_failed": {
        "en": "❌ Command execution failed: {error}",
        "zh": "❌ 命令执行失败: {error}"
    },
    "daemon_starting": {
        "en": "🏁 GPU {gpu_id} occupation daemon starting",
        "zh": "🏁 GPU {gpu_id} 占用守护开始运行"
    },
    "gpu_usage_status": {
        "en": "📊 GPU {gpu_id} current memory usage: {usage:.2f}% (Threshold: {threshold}%) - Remaining Time: {remaining_minutes}m{remaining_seconds}s",
        "zh": "📊 GPU {gpu_id} 当前显存使用率: {usage:.2f}% (阈值: {threshold}%) - 剩余时间: {remaining_minutes}分{remaining_seconds}秒"
    },
    "max_runtime_reached": {
        "en": "⏰ GPU {gpu_id} has reached the maximum runtime ({max_runtime_seconds/60} minutes), stopping...",
        "zh": "⏰ GPU {gpu_id} 已达到最大运行时间({max_runtime_seconds/60}分钟)，正在停止..."
    },
    "releasing_memory": {
        "en": "🧹 Releasing previously occupied memory on GPU {gpu_id}...",
        "zh": "🧹 释放GPU {gpu_id} 之前占用的显存..."
    },
    "memory_released": {
        "en": "✅ GPU {gpu_id} memory has been released! Current memory usage: {usage:.2f}%",
        "zh": "✅ GPU {gpu_id} 显存已释放! 当前显存使用率: {usage:.2f}%"
    },
    "process_terminating": {
        "en": "📢 Terminating the process (PID: {pid}) running on GPU {gpu_id}...",
        "zh": "📢 正在终止在GPU {gpu_id} 上运行的进程(PID: {pid})..."
    },
    "process_terminated": {
        "en": "✅ Process on GPU {gpu_id} has been terminated",
        "zh": "✅ GPU {gpu_id} 上的进程已终止"
    },
    "process_termination_error": {
        "en": "⚠️ Error terminating process: {error}",
        "zh": "⚠️ 终止进程时出错: {error}"
    },
    "cleanup_signal": {
        "en": "\n⚠️ Received termination signal or timeout, cleaning up resources...",
        "zh": "\n⚠️ 接收到终止信号或超时，正在清理资源..."
    },
    "daemon_exited": {
        "en": "👋 Multi-GPU occupation program has exited safely!",
        "zh": "👋 多GPU占用程序已安全退出!"
    },
    "multigpu_start": {
        "en": "🚀 Starting multi-GPU occupation daemon, monitoring GPUs: {gpu_ids}",
        "zh": "🚀 启动多GPU占用守护，监控GPU: {gpu_ids}"
    },
     "multigpu_exit_after":{
        "en": "⏳ Program will automatically exit after {max_runtime_minutes} minutes",
        "zh": "⏳ 程序将在 {max_runtime_minutes} 分钟后自动退出"
    },
    "multigpu_timeinfo":{
        "en": "📅 Start Time: {start_time}, Estimated End Time: {end_time}",
        "zh": "📅 开始时间: {start_time}, 预计结束时间: {end_time}"
    },
    "no_available_gpus": {
      "en": "❌ No available GPUs found, exiting program",
      "zh": "❌ 没有找到可用的GPU，退出程序"
    },
    "auto_select_gpus":{
        "en": "🔍 Automatically selected the following GPUs for occupation: {gpu_ids}",
        "zh": "🔍 自动选择了以下GPU进行占用: {gpu_ids}"
    }
}

def log_message(key, lang='zh', *args, **kwargs):
    """Logs a message from the JSON dictionary.

    Args:
        key (str): The key of the message in `json_messages`.
        lang (str): Language to use ('en' or 'zh').
        *args: Positional arguments to format the message.
        **kwargs: Keyword arguments to format the message (preferred).
    """
    if key in json_messages:
        # 根据语言选择消息
        msg = json_messages[key][lang]

        # Handle formatting with keyword arguments (preferred)
        if kwargs:
            msg = msg.format(**kwargs)
        elif args:  # Fallback to positional arguments
            try:
                msg = msg.format(*args)
            except (IndexError, KeyError) as e:
                print(f"错误: 格式化消息时出错: {e}")
                return

        print(f"{msg}")
    else:
        print(f"⚠️ Missing message key: {key}")

def get_available_gpus():
    """Gets information about all available GPUs in the system."""
    try:
        cmd = "nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits"
        output = subprocess.check_output(cmd, shell=True).decode('utf-8').strip().split('\n')

        gpus_info = []
        for line in output:
            values = line.strip().split(',')
            if len(values) >= 3:
                gpu_id = int(values[0])
                memory_used = float(values[1])
                memory_total = float(values[2])
                usage_percent = (memory_used / memory_total) * 100
                gpus_info.append({
                    'id': gpu_id,
                    'usage': usage_percent,
                    'memory_used': memory_used,
                    'memory_total': memory_total
                })

        return gpus_info
    except Exception as e:
        log_message("get_gpu_info_fail", error=e)
        return []

def select_gpus_to_occupy(num_gpus, threshold):
    """Selects GPUs to occupy."""
    available_gpus = get_available_gpus()

    # Sort by usage percentage
    available_gpus.sort(key=lambda x: x['usage'])

    # Select GPUs with usage below the threshold
    selected_gpus = [gpu['id'] for gpu in available_gpus if gpu['usage'] < threshold]

    # If no GPUs meet the criteria, select the one with the lowest usage
    if not selected_gpus and available_gpus:
        selected_gpus = [available_gpus[0]['id']]

    # Limit the number of GPUs
    selected_gpus = selected_gpus[:num_gpus]

    return selected_gpus

class GPUOccupier:
    def __init__(self, gpu_id, threshold=20, check_interval=1, occupation_ratio=0.7,
                 max_runtime_minutes=120, command=None, work_dir=None, exit_after_command=False,
                 lang='zh'):
        """
        Initializes a single GPU occupier.

        Args:
            gpu_id (int): The GPU ID to monitor.
            threshold (int): The GPU memory usage threshold (percentage) that triggers occupation.
            check_interval (int): The check interval in seconds.
            occupation_ratio (float): The memory ratio to occupy when triggered.
            max_runtime_minutes (float): The maximum runtime in minutes, after which it will automatically stop.
            command (str): The command to execute.
            work_dir (str): The working directory for command execution.
            exit_after_command (bool): Whether to exit after the command is finished.
            lang (str): Language to use for messages ('en' or 'zh').
        """
        self.gpu_id = gpu_id
        self.threshold = threshold
        self.check_interval = check_interval
        self.occupation_ratio = occupation_ratio
        self.max_runtime_seconds = max_runtime_minutes * 60
        self.tensors = []
        self.running = True
        self.start_time = time.time()
        self.command = command
        self.work_dir = work_dir
        self.command_executed = False
        self.exit_after_command = exit_after_command
        self.process = None  # Add this line to track the child process
        self.lang = lang

        log_message("monitoring_gpu", lang=self.lang, gpu_id=self.gpu_id)
        log_message("gpu_config", lang=self.lang,
                    gpu_id=self.gpu_id, 
                    threshold=self.threshold, 
                    check_interval=self.check_interval, 
                    occupation_percentage=self.occupation_ratio * 100)
        log_message("max_runtime", lang=self.lang, max_runtime_minutes=max_runtime_minutes)

        if self.command:
            log_message("will_execute_command", lang=self.lang, command=self.command)
            log_message("working_directory", lang=self.lang, work_dir=self.work_dir or "Current Directory")
            if self.exit_after_command:
                log_message("exit_after_command", lang=self.lang)
            else:
                log_message("continue_after_command", lang=self.lang)

    def get_gpu_memory_usage(self):
        """Gets GPU memory usage (percentage)."""
        try:
            cmd = f"nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits -i {self.gpu_id}"
            output = subprocess.check_output(cmd, shell=True).decode('utf-8').strip().split(',')
            memory_used = float(output[0])
            memory_total = float(output[1])
            usage_percent = (memory_used / memory_total) * 100
            return usage_percent, memory_total
        except Exception as e:
            log_message("get_gpu_info_fail", lang=self.lang, error=e)
            return 0, 0

    def occupy_memory(self, total_memory):
        """Occupies GPU memory."""
        log_message("occupy_memory_start", lang=self.lang, gpu_id=self.gpu_id)

        # Clean up previously allocated tensors
        self.release_memory()

        # Calculate memory to allocate (MB)
        memory_to_allocate = int(total_memory * self.occupation_ratio)

        try:
            # Allocate a large tensor on CUDA
            device = f'cuda:{self.gpu_id}'
            num_elements = int(memory_to_allocate * 1024 * 1024 / 4)
            tensor = torch.zeros(num_elements, dtype=torch.float32, device=device)
            self.tensors.append(tensor)

            # Perform some operations to ensure the tensor is actually allocated to the GPU
            tensor[0] = 1.0

            usage, _ = self.get_gpu_memory_usage()
            log_message("occupy_memory_success", lang=self.lang, gpu_id=self.gpu_id, usage=usage)

            # Execute command after successful occupation (if any)
            if self.command and not self.command_executed:
                self.execute_command()

        except Exception as e:
            log_message("occupy_memory_fail", lang=self.lang, gpu_id=self.gpu_id, error=e)

    def execute_command(self):
        """Executes the specified command."""
        if not self.command:
            return

        log_message("execute_command_start", lang=self.lang, gpu_id=self.gpu_id, command=self.command)
        try:
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)
            self.process = subprocess.Popen(
                self.command,
                shell=True,
                cwd=self.work_dir,
                env=env,
                # Output directly to the terminal
                stdout=None,
                stderr=None,
                universal_newlines=True
            )
            log_message("command_started", lang=self.lang, gpu_id=self.gpu_id, pid=self.process.pid)
            self.command_executed = True
            if self.exit_after_command:
                log_message("command_exited", lang=self.lang)
                self.release_memory()  #Ensure we clean up
                self.running = False  # Exit the current occupation logic
        except Exception as e:
            log_message("command_start_failed", lang=self.lang, error=e)
            self.release_memory() #Ensure we clean up
            self.running = False

    def terminate_process(self):
        """Terminates the child process."""
        if self.process is not None:
            try:
                log_message("process_terminating", lang=self.lang, gpu_id=self.gpu_id, pid=self.process.pid)
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
                self.process.terminate()
                # Give the process some time to exit gracefully
                for _ in range(3):
                    if self.process.poll() is not None:
                        break
                    time.sleep(0.5)
                # If the process is still running, force termination
                if self.process.poll() is None:
                    self.process.kill()
                    self.process.wait()
                log_message("process_terminated", lang=self.lang, gpu_id=self.gpu_id)
            except Exception as e:
                log_message("process_termination_error", lang=self.lang, error=e)

    def release_memory(self):
        """Releases previously allocated memory."""
        if self.tensors:
            log_message("releasing_memory", lang=self.lang, gpu_id=self.gpu_id)
            self.tensors = []
            torch.cuda.empty_cache()
            usage, _ = self.get_gpu_memory_usage()
            log_message("memory_released", lang=self.lang, gpu_id=self.gpu_id, usage=usage)

    def check_timeout(self):
        """Checks for timeout."""
        elapsed_seconds = time.time() - self.start_time
        return elapsed_seconds > self.max_runtime_seconds

    def get_time_remaining(self):
        """Gets the remaining time (in seconds)."""
        elapsed_seconds = time.time() - self.start_time
        return max(0, self.max_runtime_seconds - elapsed_seconds)

    def run(self):
        """Runs the monitoring loop."""
        log_message("daemon_starting", lang=self.lang, gpu_id=self.gpu_id)
        while self.running:
            # Check for timeout
            if self.check_timeout():
                log_message("max_runtime_reached", lang=self.lang, gpu_id=self.gpu_id, max_runtime_seconds=self.max_runtime_seconds)
                self.release_memory()
                self.running = False
                continue

            # Get remaining time
            time_remaining = self.get_time_remaining()
            remaining_minutes = int(time_remaining / 60)
            remaining_seconds = int(time_remaining % 60)

            usage, total_memory = self.get_gpu_memory_usage()
            log_message("gpu_usage_status", lang=self.lang, gpu_id=self.gpu_id, usage=usage, threshold=self.threshold, remaining_minutes=remaining_minutes, remaining_seconds=remaining_seconds)

            # Occupy memory only if there are no tensors and usage is below the threshold
            if usage < self.threshold and not self.tensors:
                self.occupy_memory(total_memory)

            sleep_time = min(self.check_interval, time_remaining + 0.1)
            if sleep_time > 0:
                time.sleep(sleep_time)

class MultiGPUOccupier:
    def __init__(self, gpu_ids, threshold=20, check_interval=5, occupation_ratio=0.7,
                 max_runtime_minutes=2, command=None, work_dir=None, exit_after_command=False, 
                 lang='zh'):
        """
        Initializes the multi-GPU occupation manager.
        """
        self.lang = lang
        self.gpu_occupiers = {}
        self.threads = {}
        self.running = True
        self.start_time = time.time()
        self.max_runtime_seconds = max_runtime_minutes * 60

        # Create an occupier and thread for each GPU
        for gpu_id in gpu_ids:
            self.gpu_occupiers[gpu_id] = GPUOccupier(
                gpu_id=gpu_id,
                threshold=threshold,
                check_interval=check_interval,
                occupation_ratio=occupation_ratio,
                max_runtime_minutes=max_runtime_minutes,
                command=command,
                work_dir=work_dir,
                exit_after_command=exit_after_command,
                lang=lang
            )

        # Register signal handling functions
        signal.signal(signal.SIGINT, self.cleanup)
        signal.signal(signal.SIGTERM, self.cleanup)

    def start(self):
        """Starts all GPU occupation threads."""
        log_message("multigpu_start", lang=self.lang, gpu_ids=list(self.gpu_occupiers.keys()))
        log_message("multigpu_exit_after", lang=self.lang, max_runtime_minutes=self.max_runtime_seconds/60)

        start_time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        end_time = datetime.datetime.now() + datetime.timedelta(seconds=self.max_runtime_seconds)
        end_time_str = end_time.strftime("%Y-%m-%d %H:%M:%S")
        log_message("multigpu_timeinfo", lang=self.lang, start_time = start_time_str, end_time = end_time_str)

        for gpu_id, occupier in self.gpu_occupiers.items():
            self.threads[gpu_id] = threading.Thread(target=occupier.run)
            self.threads[gpu_id].daemon = True
            self.threads[gpu_id].start()

    def cleanup(self, signum=None, frame=None):
        """Handles termination signals, cleaning up resources."""
        log_message("cleanup_signal", lang=self.lang)
        for gpu_id, occupier in self.gpu_occupiers.items():
            # Terminate child processes first
            occupier.terminate_process()
            # Then release GPU resources
            occupier.running = False
            occupier.release_memory()

        self.running = False
        log_message("daemon_exited", lang=self.lang)
        sys.exit(0)

    def check_timeout(self):
        """Checks if all GPUs have timed out."""
        return all(not occupier.running for occupier in self.gpu_occupiers.values())

    def join(self):
        """Waits for all threads to complete."""
        try:
            # The main thread keeps running until an interrupt signal is received or all GPUs have timed out
            while self.running:
                # Check if all GPUs have stopped running (including due to command completion)
                if self.check_timeout():
                    self.cleanup()
                    break  # Ensure exiting the loop
                time.sleep(1)
        except KeyboardInterrupt:
            self.cleanup()

def main():
    parser = argparse.ArgumentParser(description="Multi-functional GPU occupation daemon")
    parser.add_argument("--gpus", type=str, default=None, help="List of GPU IDs to monitor, separated by commas (e.g., '0,1,3')")
    parser.add_argument("--total-gpus", type=int, default=None, help="The total number of GPUs in the system")
    parser.add_argument("--num-gpus", type=int, default=1, help="The number of GPUs to occupy (defaults to 1)")
    parser.add_argument("--threshold", type=float, default=20, help="The memory usage threshold (percentage) that triggers occupation")
    parser.add_argument("--interval", type=float, default=5, help="The check interval (seconds)")
    parser.add_argument("--ratio", type=float, default=0.7, help="The memory ratio to occupy when triggered")
    parser.add_argument("--timeout", type=float, default=2, help="The maximum runtime in minutes, after which it will automatically stop")
    parser.add_argument("--command", type=str, help="The command to execute")
    parser.add_argument("--work-dir", type=str, help="The working directory for command execution")
    parser.add_argument("--exit-after-command", action="store_true", help="Exit the occupation after the command is finished")
    parser.add_argument("--lang", type=str, choices=['en', 'zh'], default='zh',
                      help="Language for output messages (en/zh)")

    args = parser.parse_args()

    # Parse GPU ID list
    gpu_ids = []
    if args.gpus:
        # If GPU IDs are explicitly specified, use them
        gpu_ids = [int(gpu_id.strip()) for gpu_id in args.gpus.split(",")]
    else:
        # Automatically select GPUs
        gpu_ids = select_gpus_to_occupy(args.num_gpus, args.threshold)

        if not gpu_ids:
            log_message("no_available_gpus", lang=args.lang)
            sys.exit(1)
        log_message("auto_select_gpus", lang=args.lang, gpu_ids = gpu_ids)

    # Specify process group ID in the subprocess.Popen call so child processes run in their own process group
    os.setpgrp()  # Let the main program run in its own process group

    # Create and start the multi-GPU occupier
    occupier = MultiGPUOccupier(
        gpu_ids=gpu_ids,
        threshold=args.threshold,
        check_interval=args.interval,
        occupation_ratio=args.ratio,
        max_runtime_minutes=args.timeout,
        command=args.command,
        work_dir=args.work_dir,
        exit_after_command=args.exit_after_command,
        lang=args.lang
    )

    occupier.start()
    occupier.join()

if __name__ == "__main__":
    main()