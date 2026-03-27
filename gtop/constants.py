DEFAULT_TIMEOUT = 120

SACCT_COMMAND = (
    "sacct -a -X -n -P --state=RUNNING "
    "--format=User%10,Jobid%10,JobName%20,State,Partition%40,NodeList%40,AllocTRES%120,TimeLimit%12 "
    "--units=G"
)
JOBS_SACCT_COMMAND = (
    "sacct -a -X -n -P "
    "--format=User,Jobid,JobName,State,Partition,NodeList,AllocTRES,TimeLimit "
    "--units=G"
)
GTOP_COMMAND = SACCT_COMMAND
JOBS_DEFAULT_STATES = ("RUNNING", "PENDING", "REQUEUED")

SINFO_COMMAND = (
    "sinfo -N -O "
    "nodehost:100,features:200,gres:256,gresused:256,cpusstate:100,allocmem:100,memory:100 "
    "-h"
)

SINFO_FIELD_WIDTHS = [100, 200, 256, 256, 100, 100, 100]

RESOURCE_NAMES = ("cpu", "gpu", "mem")
JOB_RESOURCE_NAMES = (*RESOURCE_NAMES, "shard")
PARTITIONS = ("priority", "gpu", "default")
SORT_CHOICES = ("name", "feature", "free-gpu", "used-gpu", "free-shard")

EXIT_SUCCESS = 0
EXIT_COMMAND_ERROR = 1
EXIT_NO_MATCHES = 2
EXIT_PARSE_ERROR = 3
