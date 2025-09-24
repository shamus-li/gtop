# gtop
Display stats about GPU usage on a SLURM cluster. Designed specifically for Cornell University's G2 and Unicorn clusters.

## Installation

```
git clone git@github.com:shamus-li/gtop.git
cd gtop
pip install .
```

If you did the above instructions in a virtual environment, you can create a symlink from the gtop executable to a directory in your PATH. While the environment is still active, use the which command to find the exact location of the installed script.
```
which gtop
mkdir -p ~/.local/bin
echo "export PATH="$HOME/.local/bin:$PATH" >> ~/.bashrc
source ~/.bashrc
ln -s GTOP_PATH ~/.local/bin/gtop   # replace GTOP_PATH with the output of `which gtop`
```

## Usage
(P/D/I) refers to compute resources that are used by a priority partition user, used by a default partition user, and idle resources.

```
gtop                    # display stats about all nodes in the cluster
gtop --gpu-only         # display stats only about nodes with gpus
gtop --disp-users       # include information which which specific jobs are using which resources
gtop --users USER       # filter the results for specific user accounts
gtop --mig-info         # display detailed MIG (Multi-Instance GPU) information
gtop --no-parallel      # disable parallel command execution (for debugging)
gtop --debug            # enable debug output for troubleshooting usage calculation issues
```

## New Features

### Sharded GPU Support
The tool now supports SLURM's sharded GPU functionality, allowing multiple jobs to share GPU resources:
- Detects and displays the specific GPU type being sharded (e.g., `4 x a100 (Sharded)`)
- Shows shard utilization with GPU type indicator: `priority/default/idle (S:GPU_TYPE)`
- Falls back to generic display for unknown shard types: `priority/default/idle (S)`

### MIG (Multi-Instance GPU) Support
Enhanced support for NVIDIA MIG partitioned GPUs:
- Automatically detects MIG instances (e.g., `1g.5gb`, `2g.10gb`, `3g.20gb`, etc.)
- Displays MIG information in GPU type column as `MIG(instance_types)`
- Use `--mig-info` flag to see detailed MIG instance breakdown per server
- Shows MIG usage as `used/idle (MIG)` format

## Troubleshooting

### Usage Shows All Zeros
If you see all nodes showing 0 usage when you know there are running jobs, try these steps:

1. **Enable debug mode**: `gtop --debug` to see detailed job processing information
2. **Check job filtering**: The tool only shows RUNNING jobs - verify jobs are in RUNNING state
3. **Verify node names match**: Debug output will show if job nodes aren't found in the server list
4. **Use sequential mode**: `gtop --no-parallel --debug` for easier troubleshooting

### Common Issues
- **Permission errors**: Ensure you have access to `sinfo` and `sacct` commands
- **Timeout errors**: Some clusters may need longer timeouts for large job lists
- **Node name mismatches**: Job node names must exactly match `sinfo` node names