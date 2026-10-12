from typing import Optional

from gtop.constants import FIELD_DELIMITER


def sacct_row(
    tres: str,
    *,
    user: str = "alice",
    job_id: str = "1",
    name: str = "train",
    state: str = "RUNNING",
    partition: str = "gpu",
    nodelist: str = "node-1",
    constraints: str = "",
    time_limit: str = "1:00:00",
    elapsed: str = "00:00:01",
    requested: str = "",
    reason: str = "",
) -> str:
    """One line of gtop's sacct output (see SACCT_FORMAT)."""
    return FIELD_DELIMITER.join(
        (
            user,
            job_id,
            name,
            state,
            partition,
            nodelist,
            constraints,
            tres,
            time_limit,
            elapsed,
            requested,
            reason,
        )
    )


def node_row(
    name: str,
    gres: str = "",
    gres_used: str = "",
    *,
    features: str = "",
    cpus: int = 0,
    alloc_cpus: int = 0,
    effective_cpus: Optional[int] = None,
    memory: int = 0,
    alloc_mem: int = 0,
    state: str = "MIXED",
    partitions: str = "",
    reason: str = "",
) -> str:
    """One line of `scontrol show node -o -d` (see NODES_COMMAND)."""
    fields = [
        f"NodeName={name}",
        f"CPUAlloc={alloc_cpus}",
        f"CPUEfctv={cpus if effective_cpus is None else effective_cpus}",
        f"CPUTot={cpus}",
        f"AvailableFeatures={features or '(null)'}",
        f"Gres={gres or '(null)'}",
    ]
    if gres_used:
        fields.append(f"GresUsed={gres_used}")
    fields += [f"RealMemory={memory}", f"AllocMem={alloc_mem}", f"State={state}"]
    if partitions:
        fields.append(f"Partitions={partitions}")
    if reason:
        fields.append(f"Reason={reason}")
    return " ".join(fields)
