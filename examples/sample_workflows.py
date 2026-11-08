"""
Sample workflows demonstrating Cirrus Slurm test harness usage.

These examples show common patterns for submitting jobs and calculating charges.
"""

from cirrus_harness import JobSubmitter, SlurmJobConfig, ChargeCalculator


def example_serial_job():
    """Submit a simple serial job."""
    print("=== Serial Job Example ===\n")
    
    config = SlurmJobConfig(
        partition="standard",
        qos="standard",
        account="z19",
        nodes=1,
        tasks_per_node=1,
        cpus_per_task=1,
        time="00:10:00",
        job_name="serial_example",
    )
    
    submitter = JobSubmitter()
    submission = submitter.submit_job(
        config,
        commands=["echo 'Hello from serial job'"],
        job_type="serial"
    )
    
    print(f"Submitted job {submission.job_id}")
    print(f"Script: {submission.script_path}")

    submitter.wait_for_completion(submission, timeout=600, verbose=True)

    # Calculate charge
    calculator = ChargeCalculator()
    charge = calculator.compute_charge(submission.job_id, config)
    print(charge)

    
    return submission


def example_mpi_job():
    """Submit an MPI job across multiple nodes."""
    print("=== MPI Job Example ===\n")
    
    config = SlurmJobConfig(
        partition="standard",
        qos="standard",
        account="t01-test",
        nodes=2,
        tasks_per_node=288,
        cpus_per_task=1,
        time="00:20:00",
        exclusive=True,
        job_name="mpi_example",
    )
    
    submitter = JobSubmitter()
    submission = submitter.submit_job(
        config,
        commands=["./my_mpi_app.x"],
        job_type="mpi"
    )
    
    print(f"Submitted job {submission.job_id}")
    print(f"Total cores: {config.total_cores}")
    print(f"Effective cores (for charging): {config.effective_cores}")
    
    return submission


def example_hybrid_job():
    """Submit a hybrid MPI+OpenMP job."""
    print("=== Hybrid MPI+OpenMP Job Example ===\n")
    
    config = SlurmJobConfig(
        partition="standard",
        qos="standard",
        account="t01-test",
        nodes=1,
        tasks_per_node=24,
        cpus_per_task=12,
        time="00:30:00",
        exclusive=True,
        job_name="hybrid_example",
    )
    
    submitter = JobSubmitter()
    submission = submitter.submit_job(
        config,
        commands=["./my_hybrid_app.x"],
        job_type="hybrid"
    )
    
    print(f"Submitted job {submission.job_id}")
    print(f"MPI ranks per node: {config.tasks_per_node}")
    print(f"OpenMP threads per rank: {config.cpus_per_task}")
    print(f"Total cores: {config.total_cores}")
    
    return submission


def example_job_array():
    """Submit a job array."""
    print("=== Job Array Example ===\n")
    
    config = SlurmJobConfig(
        partition="standard",
        qos="standard",
        account="t01-test",
        nodes=1,
        tasks_per_node=1,
        cpus_per_task=1,
        time="00:10:00",
        array_spec="1-10",
        job_name="array_example",
    )
    
    submitter = JobSubmitter()
    submission = submitter.submit_job(
        config,
        commands=["echo 'Array task $SLURM_ARRAY_TASK_ID'"],
        job_type="serial"
    )
    
    print(f"Submitted array job {submission.job_id}")
    print(f"Array spec: {config.array_spec}")
    
    return submission


def example_lowpriority_job():
    """Submit a low priority (free) job."""
    print("=== Low Priority Job Example ===\n")
    
    config = SlurmJobConfig(
        partition="standard",
        qos="lowpriority",
        account="t01-test",
        nodes=1,
        tasks_per_node=288,
        cpus_per_task=1,
        time="24:00:00",
        exclusive=True,
        job_name="lowpriority_example",
    )
    
    submitter = JobSubmitter()
    submission = submitter.submit_job(
        config,
        commands=["./my_app.x"],
        job_type="mpi"
    )
    
    print(f"Submitted low priority job {submission.job_id}")
    print(f"Max walltime: 24 hours")
    print(f"Charge multiplier: 0.0× (FREE!)")
    
    return submission


def example_highpriority_job():
    """Submit a high priority job."""
    print("=== High Priority Job Example ===\n")
    
    config = SlurmJobConfig(
        partition="standard",
        qos="highpriority",
        account="t01-test",
        nodes=1,
        tasks_per_node=288,
        cpus_per_task=1,
        time="00:30:00",
        exclusive=True,
        job_name="highpriority_example",
    )
    
    submitter = JobSubmitter()
    submission = submitter.submit_job(
        config,
        commands=["./my_app.x"],
        job_type="mpi"
    )
    
    print(f"Submitted high priority job {submission.job_id}")
    print(f"Charge multiplier: 1.5× (50% surcharge)")
    print(f"Elevated scheduling priority")
    
    return submission


def example_charge_comparison():
    """Compare charges for different configurations."""
    print("=== Charge Comparison Example ===\n")
    
    calculator = ChargeCalculator()
    
    # Example 1: Serial job, 1 core, 1 hour
    print("1. Serial job (1 core, 1 hour):")
    print("   Base charge: 1.0 core-hour")
    print()
    
    # Example 2: Full node MPI, 288 cores, 1 hour
    print("2. Full node MPI (288 cores, 1 hour):")
    print("   Base charge: 288.0 core-hours")
    print()
    
    # Example 3: Same with highpriority QoS
    print("3. Full node MPI with highpriority (288 cores, 1 hour):")
    print("   Base charge: 288.0 core-hours")
    print("   QoS multiplier: 1.5×")
    print("   Final charge: 432.0 core-hours")
    print()
    
    # Example 4: Same with lowpriority QoS
    print("4. Full node MPI with lowpriority (288 cores, 1 hour):")
    print("   Base charge: 288.0 core-hours")
    print("   QoS multiplier: 0.0×")
    print("   Final charge: 0.0 core-hours (FREE!)")
    print()
    
    # Example 5: Exclusive with partial usage
    print("5. Exclusive with 1 core requested (1 hour):")
    print("   Requested: 1 core")
    print("   Charged: 288 cores (full node)")
    print("   Base charge: 288.0 core-hours")
    print()


def example_hetjob():
    """Submit a heterogeneous job."""
    print("=== Heterogeneous Job Example ===\n")
    
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
            # Component 0: MPI simulation
            HetJobComponent(
                nodes=2,
                tasks_per_node=288,
                cpus_per_task=1,
                partition="standard",
                qos="standard",
            ),
            # Component 1: Serial analysis
            HetJobComponent(
                nodes=1,
                tasks_per_node=1,
                cpus_per_task=1,
                partition="standard",
                qos="standard",
            ),
        ],
        job_name="hetjob_example",
    )
    
    submitter = JobSubmitter()
    submission = submitter.submit_hetjob(
        hetjob_config,
        component_commands=[
            ["./mpi_simulation.x"],
            ["./serial_analysis.x"],
        ]
    )
    
    print(f"Submitted hetjob {submission.het_job_id}")
    print(f"Components: {len(hetjob_config.components)}")
    print(f"  - Component 0: {hetjob_config.components[0].nodes} nodes, MPI")
    print(f"  - Component 1: {hetjob_config.components[1].nodes} nodes, Serial")
    
    return submission


def main():
    """Run all examples."""
    print("Cirrus Slurm Test Harness - Example Workflows\n")
    print("=" * 60)
    print()
    
    # Note: These examples will fail without Cirrus access
    # Uncomment to run on Cirrus login node:
    
    example_serial_job()
    # example_mpi_job()
    # example_hybrid_job()
    # example_job_array()
    # example_lowpriority_job()
    # example_highpriority_job()
    # example_hetjob()
    
    example_charge_comparison()
    
    print("=" * 60)
    print("\nTo run actual job submissions, execute this script on Cirrus:")
    print("  python examples/sample_workflows.py")


if __name__ == "__main__":
    main()
