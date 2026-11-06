"""
Validation utilities for Slurm job configurations.
"""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


# Valid partition/QoS combinations on Cirrus
VALID_PARTITION_QOS_COMBINATIONS = {
    ("standard", "standard"): {"max_time_hours": 48, "max_nodes": 128, "max_jobs": 256},
    ("standard", "highpriority"): {"max_time_hours": 48, "max_nodes": 128, "max_jobs": 256, "multiplier": 1.5},
    ("standard", "long"): {"max_time_hours": 96, "max_nodes": 16, "max_jobs": 128},
    ("standard", "short"): {"max_time_hours": 0.33, "max_nodes": 2, "max_jobs": 2},  # 20 minutes
    ("standard", "lowpriority"): {"max_time_hours": 24, "max_nodes": 64, "max_jobs": 100, "multiplier": 0.0},
    ("standard", "largescale"): {"max_time_hours": 212, "max_nodes": 640, "max_jobs": 4, "min_nodes": 129},
    ("standard", "reservation"): {"max_time_hours": None, "max_nodes": None, "max_jobs": None},
    ("highmem", "standard"): {"max_time_hours": 48, "max_nodes": 128, "max_jobs": 256},
    ("highmem", "highpriority"): {"max_time_hours": 48, "max_nodes": 128, "max_jobs": 256, "multiplier": 1.5},
    ("highmem", "long"): {"max_time_hours": 96, "max_nodes": 16, "max_jobs": 128},
    ("highmem", "reservation"): {"max_time_hours": None, "max_nodes": None, "max_jobs": None},
}

# Cirrus hardware specifications
CIRRAS_SPECS = {
    "cores_per_node": 288,
    "standard_memory_gb": 768,
    "highmem_memory_gb": 1536,
    "sockets_per_node": 2,
    "cores_per_socket": 144,
}


@dataclass
class ValidationResult:
    """Result of a validation check."""
    valid: bool
    message: str
    warnings: List[str]


def validate_partition_qos(partition: str, qos: str) -> ValidationResult:
    """
    Validate that a partition/QoS combination is valid on Cirrus.
    
    Args:
        partition: Partition name (e.g., 'standard', 'highmem')
        qos: QoS name (e.g., 'standard', 'highpriority', 'long')
    
    Returns:
        ValidationResult with validation status and messages
    """
    key = (partition.lower(), qos.lower())
    
    if key not in VALID_PARTITION_QOS_COMBINATIONS:
        return ValidationResult(
            valid=False,
            message=f"Invalid partition/QoS combination: {partition}/{qos}",
            warnings=[]
        )
    
    config = VALID_PARTITION_QOS_COMBINATIONS[key]
    warnings = []
    
    if qos == "lowpriority":
        warnings.append("Low priority jobs require UKRI project access and at least 1 coreh in budget")
    if qos == "largescale":
        warnings.append("Largescale QoS requires minimum 129 nodes")
    if qos == "reservation":
        warnings.append("Reservation QoS requires a valid reservation ID")
    if qos == "highpriority":
        warnings.append("High priority jobs are charged at 1.5× the normal rate")
    
    return ValidationResult(
        valid=True,
        message=f"Valid combination: {partition}/{qos}",
        warnings=warnings
    )


def validate_time_format(time_str: str) -> ValidationResult:
    """
    Validate Slurm time format (HH:MM:SS or D-HH:MM:SS).
    
    Args:
        time_str: Time string in Slurm format
    
    Returns:
        ValidationResult with validation status
    """
    import re
    
    # Match D-HH:MM:SS or HH:MM:SS or MM:SS or SS
    pattern = r'^((\d+)-)?(\d{1,2}):(\d{2}):(\d{2})$'
    match = re.match(pattern, time_str.strip())
    
    if not match:
        return ValidationResult(
            valid=False,
            message=f"Invalid time format: {time_str}. Expected HH:MM:SS or D-HH:MM:SS",
            warnings=[]
        )
    
    days = int(match.group(2)) if match.group(1) else 0
    hours = int(match.group(3))
    minutes = int(match.group(4))
    seconds = int(match.group(5))
    
    if hours > 23 or minutes > 59 or seconds > 59:
        return ValidationResult(
            valid=False,
            message=f"Invalid time values in: {time_str}",
            warnings=[]
        )
    
    total_hours = days * 24 + hours + minutes / 60 + seconds / 3600
    
    warnings = []
    if total_hours > 24 * 4:  # More than 4 days
        warnings.append("Time exceeds maximum for most QoS (4 days)")
    
    return ValidationResult(
        valid=True,
        message=f"Valid time format: {time_str} ({total_hours:.2f} hours)",
        warnings=warnings
    )


def parse_time_to_seconds(time_str: str) -> int:
    """
    Parse Slurm time format to seconds.
    
    Args:
        time_str: Time string in Slurm format
    
    Returns:
        Total seconds
    """
    import re
    
    pattern = r'^((\d+)-)?(\d{1,2}):(\d{2}):(\d{2})$'
    match = re.match(pattern, time_str.strip())
    
    if not match:
        raise ValueError(f"Invalid time format: {time_str}")
    
    days = int(match.group(2)) if match.group(1) else 0
    hours = int(match.group(3))
    minutes = int(match.group(4))
    seconds = int(match.group(5))
    
    return days * 86400 + hours * 3600 + minutes * 60 + seconds


def validate_job_config(
    partition: str,
    qos: str,
    nodes: int,
    tasks_per_node: int,
    cpus_per_task: int,
    time: str,
    exclusive: bool = False,
) -> ValidationResult:
    """
    Validate a complete job configuration.
    
    Args:
        partition: Partition name
        qos: QoS name
        nodes: Number of nodes
        tasks_per_node: MPI tasks per node
        cpus_per_task: CPUs per task (for OpenMP threads)
        time: Wall time in Slurm format
        exclusive: Whether exclusive node access is requested
    
    Returns:
        ValidationResult with comprehensive validation
    """
    warnings = []
    errors = []
    
    # Validate partition/QoS
    pq_result = validate_partition_qos(partition, qos)
    if not pq_result.valid:
        errors.append(pq_result.message)
        return ValidationResult(valid=False, message="; ".join(errors), warnings=warnings)
    
    warnings.extend(pq_result.warnings)
    
    # Validate time format
    time_result = validate_time_format(time)
    if not time_result.valid:
        errors.append(time_result.message)
        return ValidationResult(valid=False, message="; ".join(errors), warnings=warnings)
    
    # Validate node count against QoS limits
    pq_config = VALID_PARTITION_QOS_COMBINATIONS[(partition.lower(), qos.lower())]
    
    if "min_nodes" in pq_config and nodes < pq_config["min_nodes"]:
        errors.append(f"QoS {qos} requires minimum {pq_config['min_nodes']} nodes, got {nodes}")
    
    if pq_config.get("max_nodes") and nodes > pq_config["max_nodes"]:
        errors.append(f"QoS {qos} allows maximum {pq_config['max_nodes']} nodes, got {nodes}")
    
    # Validate cores per node
    cores_per_node = tasks_per_node * cpus_per_task
    max_cores = CIRRAS_SPECS["cores_per_node"]
    
    if cores_per_node > max_cores:
        errors.append(f"Requested {cores_per_node} cores per node exceeds maximum {max_cores}")
    
    if exclusive and cores_per_node < max_cores:
        warnings.append(
            f"--exclusive specified but only using {cores_per_node}/{max_cores} cores. "
            f"You will be charged for all {max_cores} cores per node."
        )
    
    # Multi-node jobs require exclusive
    if nodes > 1 and not exclusive:
        warnings.append("Multi-node jobs should use --exclusive for proper resource allocation")
    
    # Validate account (basic check)
    # In practice, this would check against sacctmgr
    
    if errors:
        return ValidationResult(valid=False, message="; ".join(errors), warnings=warnings)
    
    return ValidationResult(
        valid=True,
        message=f"Valid job configuration: {nodes} nodes, {cores_per_node} cores/node, {time}",
        warnings=warnings
    )


def get_all_valid_combinations() -> List[Tuple[str, str]]:
    """
    Get all valid partition/QoS combinations.
    
    Returns:
        List of (partition, qos) tuples
    """
    return list(VALID_PARTITION_QOS_COMBINATIONS.keys())


def get_qos_multiplier(qos: str) -> float:
    """
    Get the charge multiplier for a QoS.
    
    Args:
        qos: QoS name
    
    Returns:
        Charge multiplier (1.0 = normal, 1.5 = 50% surcharge, 0.0 = free)
    """
    qos_lower = qos.lower()
    
    multipliers = {
        "standard": 1.0,
        "long": 1.0,
        "short": 1.0,
        "largescale": 1.0,
        "highpriority": 1.5,
        "lowpriority": 0.0,
        "reservation": 1.5,
    }
    
    return multipliers.get(qos_lower, 1.0)
