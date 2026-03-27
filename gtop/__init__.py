from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "build_parser": ("gtop.cli", "build_parser"),
    "cli_main": ("gtop.cli", "cli_main"),
    "main": ("gtop.cli", "main"),
    "ClusterParseError": ("gtop.collector", "ClusterParseError"),
    "CollectionOptions": ("gtop.collector", "CollectionOptions"),
    "CommandExecutionError": ("gtop.collector", "CommandExecutionError"),
    "NoMatchingServersError": ("gtop.collector", "NoMatchingServersError"),
    "collect_cluster_state": ("gtop.collector", "collect_cluster_state"),
    "DEFAULT_TIMEOUT": ("gtop.constants", "DEFAULT_TIMEOUT"),
    "EXIT_COMMAND_ERROR": ("gtop.constants", "EXIT_COMMAND_ERROR"),
    "EXIT_NO_MATCHES": ("gtop.constants", "EXIT_NO_MATCHES"),
    "EXIT_PARSE_ERROR": ("gtop.constants", "EXIT_PARSE_ERROR"),
    "EXIT_SUCCESS": ("gtop.constants", "EXIT_SUCCESS"),
    "GTOP_COMMAND": ("gtop.constants", "GTOP_COMMAND"),
    "JOBS_SACCT_COMMAND": ("gtop.constants", "JOBS_SACCT_COMMAND"),
    "JOB_RESOURCE_NAMES": ("gtop.constants", "JOB_RESOURCE_NAMES"),
    "PARTITIONS": ("gtop.constants", "PARTITIONS"),
    "RESOURCE_NAMES": ("gtop.constants", "RESOURCE_NAMES"),
    "SACCT_COMMAND": ("gtop.constants", "SACCT_COMMAND"),
    "SINFO_COMMAND": ("gtop.constants", "SINFO_COMMAND"),
    "SINFO_FIELD_WIDTHS": ("gtop.constants", "SINFO_FIELD_WIDTHS"),
    "SORT_CHOICES": ("gtop.constants", "SORT_CHOICES"),
    "ClusterState": ("gtop.models", "ClusterState"),
    "ClusterSummary": ("gtop.models", "ClusterSummary"),
    "CpuInfo": ("gtop.models", "CpuInfo"),
    "GpuInfo": ("gtop.models", "GpuInfo"),
    "JobAllocation": ("gtop.models", "JobAllocation"),
    "JobRecord": ("gtop.models", "JobRecord"),
    "JobUsage": ("gtop.models", "JobUsage"),
    "MemoryInfo": ("gtop.models", "MemoryInfo"),
    "ResourceUsageSplit": ("gtop.models", "ResourceUsageSplit"),
    "ServerState": ("gtop.models", "ServerState"),
    "TopUserSummary": ("gtop.models", "TopUserSummary"),
    "UserSummary": ("gtop.models", "UserSummary"),
    "to_jsonable": ("gtop.models", "to_jsonable"),
    "build_json_payload": ("gtop.render", "build_json_payload"),
    "format_resource": ("gtop.render", "format_resource"),
    "print_summary": ("gtop.render", "print_summary"),
    "render_table": ("gtop.render", "render_table"),
    "sort_server_names": ("gtop.render", "sort_server_names"),
    "visible_servers": ("gtop.render", "visible_servers"),
    "CommandResult": ("gtop.runner", "CommandResult"),
    "CommandRunner": ("gtop.runner", "CommandRunner"),
    "SubprocessRunner": ("gtop.runner", "SubprocessRunner"),
    "run_commands": ("gtop.runner", "run_commands"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    try:
        module_name, attr_name = _EXPORTS[name]
    except KeyError as error:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from error
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value
