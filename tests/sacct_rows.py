from gtop.constants import SACCT_DELIMITER


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
    return SACCT_DELIMITER.join(
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
