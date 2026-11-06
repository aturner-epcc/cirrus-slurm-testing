# Cirrus Slurm Test Harness

A Python test harness for submitting Slurm batch jobs on Cirrus HPC and computing job charges based on core-hours consumed.

## Overview

This test harness provides:

- **Job Submission**: Generate and submit Slurm batch scripts for various job types
- **Charge Calculation**: Compute core-hours consumed based on actual runtime and resource usage
- **Comprehensive Testing**: Test suites for all valid partition/QoS combinations, job types, and advanced features
- **Documentation**: Complete documentation with examples and best practices

## Features

### Supported Job Types

- **Serial Jobs**: Single-core execution
- **MPI Jobs**: Pure distributed-memory parallel jobs
- **Hybrid Jobs**: MPI + OpenMP mixed-mode parallel jobs
- **OpenMP Jobs**: Shared-memory parallel jobs
- **Job Arrays**: Collections of similar jobs with different parameters
- **Heterogeneous Jobs**: Multiple job components with different resource requirements

### Supported Partitions and QoS

| Partition | QoS | Max Walltime | Max Nodes | Charge Multiplier | Notes |
|-----------|-----|--------------|-----------|-------------------|-------|
| standard | standard | 48 hours | 128 | 1.0× | Default |
| standard | highpriority | 48 hours | 128 | 1.5× | 50% surcharge |
| standard | long | 4 days | 16 | 1.0× | Extended runtime |
| standard | short | 20 minutes | 2 | 1.0× | Quick turnaround |
| standard | lowpriority | 24 hours | 64 | 0.0× | Free (UKRI only) |
| standard | largescale | 212 hours | 640 | 1.0× | Min 129 nodes |
| standard | reservation | No limit | No limit | 1.5× | Requires reservation |
| highmem | standard | 48 hours | 128 | 1.0× | High memory nodes |
| highmem | highpriority | 48 hours | 128 | 1.5× | High memory + priority |
| highmem | long | 4 days | 16 | 1.0× | High memory + long |
| highmem | reservation | No limit | No limit | 1.5× | High memory reservation |

### Charge Calculation

Core-hours are calculated as:

```
charged_core_hours = (effective_cores × elapsed_seconds) / 3600 × qos_multiplier
```

Where:
- **effective_cores**: Actual cores used, or full node (288 cores) if `--exclusive` is specified
- **elapsed_seconds**: Actual runtime from Slurm accounting (`ElapsedRaw`)
- **qos_multiplier**: QoS-specific multiplier (0.0 for lowpriority, 1.5 for highpriority/reservation, 1.0 for others)

## Installation

### Prerequisites

- Python 3.9 or later
- Access to Cirrus HPC system
- Valid budget code (account)
- Slurm commands available (`sbatch`, `squeue`, `sacct`, `scancel`)

### Install from Source

```bash
cd cirrus-slurm-harness
pip install -e .
```

### Development Installation

```bash
pip install -e ".[dev]"
```

## Quick Start

### Basic Job Submission

```python
from cirrus_harness import JobSubmitter, SlurmJobConfig

# Configure a job
config = SlurmJobConfig(
    partition="standard",
    qos="standard",
    account="t01-test",  # Replace with your budget code
    nodes=1,
    tasks_per_node=288,
    cpus_per_task=1,
    time="00:10:00",
    exclusive=True,
    job_name="test_mpi",
)

# Submit the job
submitter = JobSubmitter()
submission = submitter.submit_job(
    config,
    commands=["./my_mpi_app.x"],
    job_type="mpi"
)

print(f"Submitted job {submission.job_id}")
```

### Wait for Completion and Calculate Charge

```python
from cirrus_harness import ChargeCalculator

# Wait for job to complete
submitter.wait_for_completion(submission, timeout=600, verbose=True)

# Calculate charge
calculator = ChargeCalculator()
charge = calculator.compute_charge(submission.job_id, config)

print(charge)
# Output:
# Job 12345:
#   State: COMPLETED (Exit: 0:0)
#   Elapsed: 600s (0.17h)
#   Cores: 288 allocated, 288 charged
#   Base core-hours: 48.00
#   QoS multiplier: 1.0×
#   Charged core-hours: 48.00
```

### Detailed Charge Explanation

```python
explanation = calculator.explain_charge(submission.job_id, config)
print(explanation)
```

## Job Types

### Serial Jobs

```python
config = SlurmJobConfig(
    partition="standard",
    qos="standard",
    account="t01-test",
    nodes=1,
    tasks_per_node=1,
    cpus_per_task=1,
    time="00:10:00",
    job_name="serial_test",
)

submission = submitter.submit_job(
    config,
    commands=["./my_serial_app.x"],
    job_type="serial"
)
```

### MPI Jobs

```python
config = SlurmJobConfig(
    partition="standard",
    qos="standard",
    account="t01-test",
    nodes=2,
    tasks_per_node=288,
    cpus_per_task=1,
    time="00:20:00",
    exclusive=True,  # Required for multi-node
    job_name="mpi_test",
)

submission = submitter.submit_job(
    config,
    commands=["./my_mpi_app.x"],
    job_type="mpi"
)
```

### Hybrid MPI+OpenMP Jobs

```python
config = SlurmJobConfig(
    partition="standard",
    qos="standard",
    account="t01-test",
    nodes=1,
    tasks_per_node=24,  # MPI ranks
    cpus_per_task=12,   # OpenMP threads per rank
    time="00:30:00",
    exclusive=True,
    job_name="hybrid_test",
)

submission = submitter.submit_job(
    config,
    commands=["./my_hybrid_app.x"],
    job_type="hybrid"
)
```

### Job Arrays

```python
config = SlurmJobConfig(
    partition="standard",
    qos="standard",
    account="t01-test",
    nodes=1,
    tasks_per_node=1,
    cpus_per_task=1,
    time="00:10:00",
    array_spec="1-10",  # 10 array tasks
    job_name="array_test",
)

submission = submitter.submit_job(
    config,
    commands=["./my_app.x $SLURM_ARRAY_TASK_ID"],
    job_type="serial"
)

# Calculate total charge for all array tasks
summary = calculator.compute_array_charge(submission.job_id, config)
print(f"Total core-hours: {summary.total_core_hours:.2f}")
```

### Heterogeneous Jobs

```python
from cirrus_harness import HetJobConfig, HetJobComponent

base_config = SlurmJobConfig(
    partition="standard",
    qos="standard",
    account="t01-test",
    time="00:30:00",
)

hetjob_config = HetJobConfig(
    base_config=base_config,
    components=[
        # Component 0: MPI job
        HetJobComponent(
            nodes=2,
            tasks_per_node=288,
            cpus_per_task=1,
            partition="standard",
            qos="standard",
        ),
        # Component 1: Serial post-processing
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

submission = submitter.submit_hetjob(
    hetjob_config,
    component_commands=[
        ["./mpi_app.x"],
        ["./serial_app.x"],
    ]
)

# Calculate per-component charges
summary = calculator.compute_hetjob_charge(submission.het_job_id)
for i, charge in enumerate(summary.component_charges):
    print(f"Component {i}: {charge.charged_core_hours:.2f} core-hours")
print(f"Total: {summary.total_core_hours:.2f} core-hours")
```

## Exclusive Access

The `--exclusive` flag reserves an entire node regardless of actual core usage. **You are charged for all 288 cores per node** even if you request fewer.

```python
# Using 1 core with --exclusive: charged for 288 cores
config_exclusive = SlurmJobConfig(
    partition="standard",
    qos="standard",
    account="t01-test",
    nodes=1,
    tasks_per_node=1,
    cpus_per_task=1,
    time="01:00:00",
    exclusive=True,  # Charges for 288 cores
)

# Using 1 core without --exclusive: charged for 1 core
config_shared = SlurmJobConfig(
    partition="standard",
    qos="standard",
    account="t01-test",
    nodes=1,
    tasks_per_node=1,
    cpus_per_task=1,
    time="01:00:00",
    exclusive=False,  # Charges for 1 core
)
```

**When to use `--exclusive`:**
- Multi-node jobs (required)
- Performance-sensitive jobs that need isolation from other users
- Jobs requiring access to full node memory

**When NOT to use `--exclusive`:**
- Single-core or small jobs where cost is a concern
- Jobs that can share nodes with other users' jobs

## Low Priority Access

Low priority jobs are **free** but only available to UKRI-funded projects:

```python
config = SlurmJobConfig(
    partition="standard",
    qos="lowpriority",  # Free!
    account="t01-test",  # Must have at least 1 coreh budget
    nodes=1,
    tasks_per_node=288,
    cpus_per_task=1,
    time="24:00:00",  # Max 24 hours
    job_name="lowpriority_test",
)
```

**Limits:**
- Maximum 64 nodes
- Maximum 24 hours walltime
- Maximum 100 concurrent jobs
- Requires UKRI project access
- Requires at least 1 coreh in budget (but not charged)

## High Priority Access

High priority jobs run with elevated priority but incur a **50% surcharge**:

```python
config = SlurmJobConfig(
    partition="standard",
    qos="highpriority",  # 1.5× charge
    account="t01-test",
    nodes=1,
    tasks_per_node=288,
    cpus_per_task=1,
    time="00:30:00",
    job_name="highpriority_test",
)
```

## Running Tests

### Unit Tests

```bash
pytest tests/ -v
```

### Integration Tests (Requires Cirrus Access)

```bash
pytest tests/ -v -m integration
```

### Specific Test Suites

```bash
# Test partition/QoS combinations
pytest tests/test_partitions_qos.py -v

# Test job types
pytest tests/test_job_types.py -v

# Test exclusive access
pytest tests/test_exclusive.py -v

# Test job arrays
pytest tests/test_arrays.py -v

# Test low priority
pytest tests/test_lowpriority.py -v

# Test heterogeneous jobs
pytest tests/test_hetjobs.py -v
```

## API Reference

### SlurmJobConfig

Configuration for a Slurm batch job.

**Parameters:**
- `partition` (str): Partition name (`standard`, `highmem`)
- `qos` (str): Quality of Service (`standard`, `highpriority`, `long`, `short`, `lowpriority`, `largescale`)
- `account` (str): Budget code to charge
- `nodes` (int): Number of nodes (default: 1)
- `tasks_per_node` (int): MPI tasks per node (default: 1)
- `cpus_per_task` (int): CPUs per task for OpenMP (default: 1)
- `time` (str): Wall time in `HH:MM:SS` format (default: `00:10:00`)
- `exclusive` (bool): Exclusive node access (default: False)
- `job_name` (str, optional): Job name
- `array_spec` (str, optional): Job array specification (e.g., `1-10`, `1-100%10`)

### JobSubmitter

Submit and manage Slurm batch jobs.

**Methods:**
- `submit_job(config, commands, job_type)`: Submit a job
- `submit_hetjob(config, component_commands)`: Submit a heterogeneous job
- `wait_for_completion(submission, timeout, verbose)`: Wait for job completion
- `cancel_job(submission)`: Cancel a job

### ChargeCalculator

Compute job charges from Slurm accounting data.

**Methods:**
- `compute_charge(job_id, config)`: Calculate charge for a single job
- `compute_array_charge(parent_job_id, config)`: Calculate charges for job array
- `compute_hetjob_charge(het_job_id)`: Calculate charges for heterogeneous job
- `explain_charge(job_id, config)`: Generate detailed charge explanation

## Best Practices

### Resource Requests

1. **Be realistic with time requests**: Shorter walltimes usually mean shorter queue waits
2. **Use `--time-min` for flexibility**: Enable backfill scheduling
3. **Request appropriate cores**: Don't over-request "just in case"
4. **Use `--exclusive` judiciously**: Only when you genuinely need node isolation

### Cost Optimization

1. **Use lowpriority for development**: Free but may wait longer in queue
2. **Avoid exclusive for small jobs**: You'll be charged for unused cores
3. **Monitor charges**: Use `explain_charge()` to understand costs
4. **Consider job arrays**: More efficient than many individual submissions

### Storage

1. **Use `/epccfs` for jobs**: Home directories not available to batch jobs
2. **Clean up after jobs**: Remove temporary files to avoid quota issues
3. **Backup important results**: Data deleted 1 month after project end

### Polling

**Never poll the queue rapidly.** Minimum polling interval is **60 seconds** to avoid disrupting the scheduler for all users.

```python
# Good
submitter.wait_for_completion(submission, poll_interval=60)

# Bad - violates policy
submitter.wait_for_completion(submission, poll_interval=5)  # Don't do this!
```

## Troubleshooting

### Common Errors

**`Invalid account`**
- Check your budget code in SAFE
- Verify account with `sacctmgr show assoc user=$USER`

**`Job violates accounting/QOS policy`**
- Check QoS limits (time, nodes, jobs)
- Verify partition/QoS combination is valid

**`PENDING: Priority`**
- Higher priority jobs are ahead in queue
- Consider using `highpriority` QoS (1.5× charge) or `lowpriority` (free, lower priority)

**`PENDING: Resources`**
- Requested resources not available
- Try smaller job or more flexible time (`--time-min`)

### Getting Help

- **Cirrus Documentation**: https://docs.cirrus.ac.uk/
- **Service Desk**: support@cirrus.ac.uk
- **Service Status**: https://www.cirrus.ac.uk/support-access/status/

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions welcome! Please read CONTRIBUTING.md for guidelines.

## Acknowledgements

This project is developed for use on the Cirrus HPC service, operated by EPCC on behalf of the UK's research community.
