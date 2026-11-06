"""
Pytest fixtures for Cirrus Slurm test harness.

Provides fixtures for job configurations, mock Slurm commands, and
test utilities.
"""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from cirrus_harness.job_submitter import SlurmJobConfig, HetJobConfig, HetJobComponent
from cirrus_harness.charge_calculator import ChargeCalculator
from cirrus_harness.utils.slurm_commands import SacctRecord


# Valid partition/QoS combinations for parameterized tests
VALID_PARTITION_QOS = [
    ("standard", "standard"),
    ("standard", "highpriority"),
    ("standard", "long"),
    ("standard", "short"),
    ("standard", "lowpriority"),
    ("standard", "largescale"),
    ("standard", "reservation"),
    ("highmem", "standard"),
    ("highmem", "highpriority"),
    ("highmem", "long"),
    ("highmem", "reservation"),
]


@pytest.fixture
def temp_output_dir():
    """Create a temporary directory for job scripts and output."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def basic_job_config():
    """Basic job configuration for testing."""
    return SlurmJobConfig(
        partition="standard",
        qos="standard",
        account="test-account",
        nodes=1,
        tasks_per_node=1,
        cpus_per_task=1,
        time="00:10:00",
    )


@pytest.fixture
def mpi_job_config():
    """MPI job configuration for testing."""
    return SlurmJobConfig(
        partition="standard",
        qos="standard",
        account="test-account",
        nodes=2,
        tasks_per_node=288,
        cpus_per_task=1,
        time="00:20:00",
        exclusive=True,
        job_name="mpi_test",
    )


@pytest.fixture
def hybrid_job_config():
    """Hybrid MPI+OpenMP job configuration."""
    return SlurmJobConfig(
        partition="standard",
        qos="standard",
        account="test-account",
        nodes=1,
        tasks_per_node=24,
        cpus_per_task=12,
        time="00:30:00",
        exclusive=True,
        job_name="hybrid_test",
    )


@pytest.fixture
def exclusive_job_config():
    """Job with exclusive node access."""
    return SlurmJobConfig(
        partition="standard",
        qos="standard",
        account="test-account",
        nodes=1,
        tasks_per_node=1,
        cpus_per_task=1,
        time="00:10:00",
        exclusive=True,
        job_name="exclusive_test",
    )


@pytest.fixture
def array_job_config():
    """Job array configuration."""
    return SlurmJobConfig(
        partition="standard",
        qos="standard",
        account="test-account",
        nodes=1,
        tasks_per_node=1,
        cpus_per_task=1,
        time="00:10:00",
        array_spec="1-5",
        job_name="array_test",
    )


@pytest.fixture
def hetjob_config():
    """Heterogeneous job configuration."""
    base_config = SlurmJobConfig(
        partition="standard",
        qos="standard",
        account="test-account",
        time="00:10:00",
    )
    
    return HetJobConfig(
        base_config=base_config,
        components=[
            HetJobComponent(
                nodes=1,
                tasks_per_node=288,
                cpus_per_task=1,
                partition="standard",
                qos="standard",
            ),
            HetJobComponent(
                nodes=1,
                tasks_per_node=1,
                cpus_per_task=1,
                partition="standard",
                qos="standard",
            ),
        ],
        job_name="hetjob_test",
    )


@pytest.fixture
def mock_sacct_record():
    """Mock SacctRecord for testing charge calculation."""
    return SacctRecord(
        job_id="12345",
        job_name="test_job",
        partition="standard",
        account="test-account",
        state="COMPLETED",
        exit_code="0:0",
        elapsed_raw=600,  # 10 minutes
        allocated_cpus=288,
        allocated_nodes=1,
        start_time="2024-01-01T10:00:00",
        end_time="2024-01-01T10:10:00",
        qos="standard",
    )


@pytest.fixture
def mock_sacct_completed():
    """Mock sacct command returning completed job data."""
    def _mock_sacct(job_id, whole_hetjob=True):
        return [
            SacctRecord(
                job_id=job_id,
                job_name="test_job",
                partition="standard",
                account="test-account",
                state="COMPLETED",
                exit_code="0:0",
                elapsed_raw=600,
                allocated_cpus=288,
                allocated_nodes=1,
                start_time="2024-01-01T10:00:00",
                end_time="2024-01-01T10:10:00",
                qos="standard",
            )
        ]
    
    with patch('cirrus_harness.charge_calculator.SlurmCLI.get_job_accounting', side_effect=_mock_sacct):
        yield _mock_sacct


@pytest.fixture
def charge_calculator():
    """Charge calculator instance."""
    return ChargeCalculator()


@pytest.fixture
def mock_sbatch():
    """Mock sbatch command."""
    with patch('cirrus_harness.utils.slurm_commands.subprocess.run') as mock_run:
        mock_run.return_value = MagicMock(
            stdout="12345\n",
            stderr="",
            returncode=0
        )
        yield mock_run


@pytest.fixture
def mock_squeue():
    """Mock squeue command."""
    def _mock_get_status(job_id):
        from cirrus_harness.utils.slurm_commands import SqueueRecord
        return SqueueRecord(
            job_id=job_id,
            partition="standard",
            name="test_job",
            user="testuser",
            state="R",
            time="0:10",
            nodes=1,
            nodelist="cs-n0001"
        )
    
    with patch('cirrus_harness.utils.slurm_commands.SlurmCLI.get_job_status', side_effect=_mock_get_status):
        yield _mock_get_status


@pytest.fixture
def integration_test_marker():
    """Marker for integration tests requiring Cirrus access."""
    return pytest.mark.integration


# Helper functions for tests

def create_mock_sacct_record(
    job_id="12345",
    elapsed_seconds=600,
    allocated_cpus=288,
    allocated_nodes=1,
    state="COMPLETED",
    qos="standard"
):
    """Helper to create mock SacctRecord."""
    return SacctRecord(
        job_id=job_id,
        job_name="test_job",
        partition="standard",
        account="test-account",
        state=state,
        exit_code="0:0",
        elapsed_raw=elapsed_seconds,
        allocated_cpus=allocated_cpus,
        allocated_nodes=allocated_nodes,
        start_time="2024-01-01T10:00:00",
        end_time="2024-01-01T10:10:00",
        qos=qos,
    )
