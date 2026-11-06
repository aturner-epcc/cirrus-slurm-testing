"""
Cirrus Slurm Test Harness

A Python test harness for submitting Slurm batch jobs on Cirrus HPC
and computing job charges based on core-hours consumed.
"""

from cirrus_harness.job_submitter import JobSubmitter, SlurmJobConfig, JobSubmission, HetJobSubmission
from cirrus_harness.charge_calculator import ChargeCalculator, JobChargeRecord
from cirrus_harness.utils.slurm_commands import SlurmCLI, SacctRecord
from cirrus_harness.utils.validators import validate_partition_qos, validate_job_config

__version__ = "0.1.0"
__all__ = [
    "JobSubmitter",
    "SlurmJobConfig",
    "JobSubmission",
    "HetJobSubmission",
    "ChargeCalculator",
    "JobChargeRecord",
    "SlurmCLI",
    "SacctRecord",
    "validate_partition_qos",
    "validate_job_config",
]
