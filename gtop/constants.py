from .runner import Command

DEFAULT_TIMEOUT = 120

# Job names and constraints may contain "|", so sacct fields use a control character.
SACCT_DELIMITER = "\x01"
# AllocTRES is empty until a job starts; ReqTRES holds what a pending job asked for.
SACCT_FORMAT = (
    "--format=User,JobID,JobName,State,Partition,NodeList,Constraints,AllocTRES,"
    "TimeLimit,Elapsed,ReqTRES,Reason"
)
SACCT_FIELD_COUNT = 12
# slurmdbd can keep a RUNNING record after a job ends. Slurm stops jobs within
# minutes of their limit (OverTimeLimit + KillWait), so anything an hour past it is stale.
STALE_RUNNING_GRACE_SECONDS = 3600

SACCT_COMMAND: Command = (
    "sacct",
    "-a",
    "-X",
    "-n",
    "-P",
    f"--delimiter={SACCT_DELIMITER}",
    "--state=RUNNING",
    SACCT_FORMAT,
    "--units=G",
)
JOBS_SACCT_COMMAND: Command = (
    "sacct",
    "-a",
    "-X",
    "-n",
    "-P",
    f"--delimiter={SACCT_DELIMITER}",
    # A window of "now" returns only active jobs; the default since-midnight
    # window returns every finished job too, and --state=PENDING takes ~40 s.
    "--starttime=now",
    "--endtime=now",
    SACCT_FORMAT,
    "--units=G",
)
ACTIVE_JOB_STATES = ("RUNNING", "PENDING", "REQUEUED")
# squeue sees jobs slurmdbd has no current record for (requeued array tasks whose
# eligible time is in the future, unexpanded pending array ranges), but only the
# jobs PrivateData lets this user see. Fields match SACCT_FORMAT, in order.
SQUEUE_FORMAT = f"{SACCT_DELIMITER},".join(
    f"{field}:0"
    for field in (
        "UserName",
        "JobArrayId",
        "Name",
        "State",
        "Partition",
        "NodeList",
        "Feature",
        "tres-alloc",
        "TimeLimit",
        "TimeUsed",
        # squeue's tres-alloc already shows requested TRES for pending jobs.
        "tres-alloc",
        "Reason",
    )
)
SQUEUE_COMMAND: Command = (
    # Without SLURM_BITSTR_LEN=0 squeue cuts array task ranges at 32 characters.
    # Absolute path: uv installs a ~/.local/bin/env script that shadows env(1).
    "/usr/bin/env",
    "SLURM_BITSTR_LEN=0",
    "squeue",
    "-h",
    "-a",
    "-t",
    ",".join(ACTIVE_JOB_STATES),
    "-O",
    SQUEUE_FORMAT,
)

SINFO_COMMAND: Command = (
    "sinfo",
    # Without -a, sinfo hides nodes that are only in partitions the user cannot use.
    "-a",
    "-N",
    "-O",
    "nodehost:100,features:200,gres:256,gresused:256,cpusstate:100,allocmem:100,"
    "memory:100,statelong:50,reason:300",
    "--exact",
    "-h",
)
SINFO_FIELD_WIDTHS = (100, 200, 256, 256, 100, 100, 100, 50, 300)

JOB_RESOURCE_NAMES = ("cpu", "gpu", "mem", "shard")
EXIT_SUCCESS = 0
EXIT_COMMAND_ERROR = 1
EXIT_NO_MATCHES = 2
EXIT_PARSE_ERROR = 3
