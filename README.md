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
gtop                # display stats about all nodes in the cluster
gtop --gpu-only     # display stats only about nodes with gpus
gtop --disp-users   # include information which which specific jobs are using which resources
gtop --users USER   # filter the results for specific user accounts
```