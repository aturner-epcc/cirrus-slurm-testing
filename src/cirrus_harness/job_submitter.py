"""
Slurm job submission logic.

Provides classes for configuring and submitting Slurm batch jobs on Cirrus.
"""

import subprocess
import tempfile
import os
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
from dataclasses import dataclass, field
import logging

from cirrus_harness.utils.validators import (
    validate_job_config,
    parse_time_to_seconds,
    CIRRAS_SPECS,
)
from cirrus_harness.utils.slurm_commands import SlurmCLI

logger = logging.getLogger(__name__)


@dataclass
class SlurmJobConfig:
    """
    Configuration for a Slurm batch job.
    
    Attributes:
        partition: Partition name (standard, highmem)
        qos: Quality of Service (standard, highpriority, long, short, lowpriority, largescale)
        account: Budget/account code to charge
        nodes: Number of nodes to request
        tasks_per_node: MPI tasks per node
        cpus_per_task: CPUs per task (for OpenMP threads)
        time: Wall time in Slurm format (HH:MM:SS)
        exclusive: Whether to request exclusive node access
        job_name: Job name (optional)
        output_file: Output file pattern (optional)
        error_file: Error file pattern (optional)
        workdir: Working directory for the job
        array_spec: Job array specification (e.g., "1-10", "1-100%10")
        reservation: Reservation ID (for reservation QoS)
        additional_options: Additional SBATCH directives
    """
    partition: str
    qos: str
    account: str
    nodes: int = 1
    tasks_per_node: int = 1
    cpus_per_task: int = 1
    time: str = "00:10:00"
    exclusive: bool = False
    job_name: Optional[str] = None
    output_file: Optional[str] = None
    error_file: Optional[str] = None
    workdir: Optional[str] = None
    array_spec: Optional[str] = None
    reservation: Optional[str] = None
    additional_options: Dict[str, str] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.job_name is None:
            self.job_name = f"test_{self.partition}_{self.qos}"
        
        if self.workdir is None:
            self.workdir = os.getcwd()
    
    @property
    def total_cores(self) -> int:
        """Total cores allocated (without exclusive adjustment)."""
        return self.nodes * self.tasks_per_node * self.cpus_per_task
    
    @property
    def effective_cores(self) -> int:
        """
        Effective cores for charging.
        
        With --exclusive, charge for entire node regardless of actual request.
        """
        if self.exclusive:
            return self.nodes * CIRRAS_SPECS["cores_per_node"]
        return self.total_cores
    
    def validate(self) -> tuple[bool, str, List[str]]:
        """
        Validate the job configuration.
        
        Returns:
            Tuple of (is_valid, message, warnings)
        """
        result = validate_job_config(
            partition=self.partition,
            qos=self.qos,
            nodes=self.nodes,
            tasks_per_node=self.tasks_per_node,
            cpus_per_task=self.cpus_per_task,
            time=self.time,
            exclusive=self.exclusive,
        )
        return result.valid, result.message, result.warnings


@dataclass
class HetJobComponent:
    """
    Configuration for one component of a heterogeneous job.
    
    Attributes:
        nodes: Number of nodes for this component
        tasks_per_node: MPI tasks per node
        cpus_per_task: CPUs per task
        partition: Partition for this component (optional, inherits from hetjob)
        qos: QoS for this component (optional, inherits from hetjob)
        constraint: Node constraints for this component
    """
    nodes: int = 1
    tasks_per_node: int = 1
    cpus_per_task: int = 1
    partition: Optional[str] = None
    qos: Optional[str] = None
    constraint: Optional[str] = None


@dataclass
class HetJobConfig:
    """
    Configuration for a heterogeneous job.
    
    Attributes:
        base_config: Base configuration (account, time, etc.)
        components: List of component configurations
        job_name: Job name
    """
    base_config: SlurmJobConfig
    components: List[HetJobComponent]
    job_name: Optional[str] = None
    
    def __post_init__(self):
        if self.job_name is None:
            self.job_name = f"hetjob_{self.base_config.partition}"


@dataclass
class JobSubmission:
    """
    Result of a job submission.
    
    Attributes:
        job_id: Slurm job ID
        config: Job configuration used
        script_path: Path to submitted script
        submit_time: Submission timestamp
    """
    job_id: str
    config: SlurmJobConfig
    script_path: Path
    submit_time: float


@dataclass
class HetJobSubmission:
    """
    Result of a heterogeneous job submission.
    
    Attributes:
        het_job_id: Heterogeneous job leader ID
        component_ids: List of component job IDs
        config: HetJob configuration used
        script_path: Path to submitted script
        submit_time: Submission timestamp
    """
    het_job_id: str
    component_ids: List[str]
    config: HetJobConfig
    script_path: Path
    submit_time: float


class JobSubmitter:
    """
    Submit and manage Slurm batch jobs on Cirrus.
    
    Example:
        >>> config = SlurmJobConfig(
        ...     partition="standard",
        ...     qos="standard",
        ...     account="t01-test",
        ...     nodes=1,
        ...     tasks_per_node=288,
        ...     time="00:10:00",
        ...     exclusive=True,
        ... )
        >>> submitter = JobSubmitter()
        >>> submission = submitter.submit_job(config)
        >>> print(f"Submitted job {submission.job_id}")
    """
    
    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize job submitter.
        
        Args:
            output_dir: Directory for job scripts and output (default: current dir)
        """
        self.output_dir = output_dir or Path.cwd()
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_script(
        self,
        config: SlurmJobConfig,
        commands: List[str],
        job_type: str = "mpi"
    ) -> Path:
        """
        Generate a Slurm batch script.
        
        Args:
            config: Job configuration
            commands: Commands to execute in the job
            job_type: Type of job (serial, mpi, hybrid, openmp)
        
        Returns:
            Path to generated script
        """
        lines = ["#!/bin/bash", ""]
        
        # SBATCH directives
        lines.append(f"#SBATCH --job-name={config.job_name}")
        lines.append(f"#SBATCH --time={config.time}")
        lines.append(f"#SBATCH --account={config.account}")
        lines.append(f"#SBATCH --partition={config.partition}")
        lines.append(f"#SBATCH --qos={config.qos}")
        
        if config.reservation:
            lines.append(f"#SBATCH --reservation={config.reservation}")
        
        # Resource specifications
        if job_type == "serial":
            lines.append(f"#SBATCH --ntasks=1")
            if config.exclusive:
                lines.append("#SBATCH --exclusive")
        else:
            lines.append(f"#SBATCH --nodes={config.nodes}")
            lines.append(f"#SBATCH --tasks-per-node={config.tasks_per_node}")
            lines.append(f"#SBATCH --cpus-per-task={config.cpus_per_task}")
            
            if config.exclusive or config.nodes > 1:
                lines.append("#SBATCH --exclusive")
        
        # Job array
        if config.array_spec:
            lines.append(f"#SBATCH --array={config.array_spec}")
            lines.append("#SBATCH --output=array_%a.out")
        
        # Output files
        if config.output_file:
            lines.append(f"#SBATCH --output={config.output_file}")
        if config.error_file:
            lines.append(f"#SBATCH --error={config.error_file}")
        
        # Additional options
        for key, value in config.additional_options.items():
            lines.append(f"#SBATCH --{key}={value}")
        
        lines.append("")
        
        # Change to working directory
        lines.append(f"cd {config.workdir}")
        lines.append("")
        
        # Environment setup and commands based on job type
        if job_type == "serial":
            lines.append("# Serial job - enforce single thread")
            lines.append("export OMP_NUM_THREADS=1")
            lines.append("")
            lines.append("# Run serial job")
            lines.append("srun " + " ".join(commands))
            lines.append("")
        elif job_type == "mpi":
            lines.append("# MPI job - enforce single thread per rank")
            lines.append("export OMP_NUM_THREADS=1")
            lines.append("")
            lines.append("# Launch MPI job")
            lines.append("srun --hint=nomultithread --distribution=block:block \\\\")
            lines.append("    " + " \\\n    ".join(commands))
            lines.append("")
        elif job_type == "hybrid":
            lines.append("# Hybrid MPI+OpenMP job")
            lines.append(f"export OMP_NUM_THREADS={config.cpus_per_task}")
            lines.append("export OMP_PLACES=cores")
            lines.append("export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK")
            lines.append("")
            lines.append("# Launch hybrid job")
            lines.append("srun --hint=nomultithread --distribution=block:block \\\\")
            lines.append("    " + " \\\n    ".join(commands))
            lines.append("")
        elif job_type == "openmp":
            lines.append("# OpenMP-only job")
            lines.append(f"export OMP_NUM_THREADS={config.cpus_per_task}")
            lines.append("export OMP_PLACES=cores")
            lines.append("export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK")
            lines.append("")
            lines.append("# Launch OpenMP job")
            lines.append("srun --hint=nomultithread --distribution=block:block \\\\")
            lines.append("    " + " \\\n    ".join(commands))
            lines.append("")
        else:
            # Generic
            for cmd in commands:
                lines.append(cmd)
            lines.append("")
        
        script_path = self.output_dir / f"{config.job_name}.sh"
        
        with open(script_path, 'w') as f:
            f.write('\n'.join(lines))
        
        # Make executable
        os.chmod(script_path, 0o755)
        
        logger.info(f"Generated job script: {script_path}")
        return script_path
    
    def generate_hetjob_script(
        self,
        config: HetJobConfig,
        component_commands: List[List[str]]
    ) -> Path:
        """
        Generate a heterogeneous job script.
        
        Args:
            config: HetJob configuration
            component_commands: Commands for each component
        
        Returns:
            Path to generated script
        """
        lines = ["#!/bin/bash", ""]
        base = config.base_config
        
        # Base SBATCH directives (component 0)
        lines.append(f"#SBATCH --job-name={config.job_name}")
        lines.append(f"#SBATCH --time={base.time}")
        lines.append(f"#SBATCH --account={base.account}")
        lines.append(f"#SBATCH --partition={base.partition}")
        lines.append(f"#SBATCH --qos={base.qos}")
        
        if base.reservation:
            lines.append(f"#SBATCH --reservation={base.reservation}")
        
        # Component 0 resources
        comp0 = config.components[0] if config.components else HetJobComponent()
        lines.append(f"#SBATCH --nodes={comp0.nodes}")
        lines.append(f"#SBATCH --tasks-per-node={comp0.tasks_per_node}")
        lines.append(f"#SBATCH --cpus-per-task={comp0.cpus_per_task}")
        
        if comp0.constraint:
            lines.append(f"#SBATCH --constraint={comp0.constraint}")
        
        lines.append("")
        
        # Additional components
        for i, comp in enumerate(config.components[1:], start=1):
            lines.append("#SBATCH hetjob")
            
            if comp.partition:
                lines.append(f"#SBATCH --partition={comp.partition}")
            if comp.qos:
                lines.append(f"#SBATCH --qos={comp.qos}")
            
            lines.append(f"#SBATCH --nodes={comp.nodes}")
            lines.append(f"#SBATCH --tasks-per-node={comp.tasks_per_node}")
            lines.append(f"#SBATCH --cpus-per-task={comp.cpus_per_task}")
            
            if comp.constraint:
                lines.append(f"#SBATCH --constraint={comp.constraint}")
            
            lines.append("")
        
        # Change to working directory
        lines.append(f"cd {base.workdir}")
        lines.append("")
        
        # Launch commands for each component
        lines.append("# Launch heterogeneous job steps")
        srun_parts = []
        for i, (comp, cmds) in enumerate(zip(config.components, component_commands)):
            for cmd in cmds:
                srun_parts.append(f"srun --het-group={i} {cmd} : \\\\")
        
        # Remove last backslash and join
        if srun_parts:
            srun_parts[-1] = srun_parts[-1].rstrip(" \\\\")
            lines.append("    ".join(srun_parts))
            lines.append("")
        
        script_path = self.output_dir / f"{config.job_name}_hetjob.sh"
        
        with open(script_path, 'w') as f:
            f.write('\n'.join(lines))
        
        os.chmod(script_path, 0o755)
        
        logger.info(f"Generated hetjob script: {script_path}")
        return script_path
    
    def submit_job(
        self,
        config: SlurmJobConfig,
        commands: Optional[List[str]] = None,
        job_type: str = "mpi",
        validate: bool = True
    ) -> JobSubmission:
        """
        Submit a Slurm batch job.
        
        Args:
            config: Job configuration
            commands: Commands to execute (default: simple hostname test)
            job_type: Type of job (serial, mpi, hybrid, openmp)
            validate: Whether to validate config before submission
        
        Returns:
            JobSubmission with job ID and metadata
        
        Raises:
            ValueError: If configuration is invalid
            RuntimeError: If submission fails
        """
        import time
        
        # Validate configuration
        if validate:
            is_valid, message, warnings = config.validate()
            if not is_valid:
                raise ValueError(f"Invalid job configuration: {message}")
            for warning in warnings:
                logger.warning(warning)
        
        # Default commands
        if commands is None:
            commands = ["hostname"]
        
        # Generate script
        script_path = self.generate_script(config, commands, job_type)
        
        # Submit job
        job_id = SlurmCLI.submit_job(script_path, account=config.account)
        
        submission = JobSubmission(
            job_id=job_id,
            config=config,
            script_path=script_path,
            submit_time=time.time()
        )
        
        logger.info(f"Submitted job {job_id}")
        return submission
    
    def submit_hetjob(
        self,
        config: HetJobConfig,
        component_commands: Optional[List[List[str]]] = None,
        validate: bool = True
    ) -> HetJobSubmission:
        """
        Submit a heterogeneous job.
        
        Args:
            config: HetJob configuration
            component_commands: Commands for each component
            validate: Whether to validate config before submission
        
        Returns:
            HetJobSubmission with job IDs and metadata
        """
        import time
        
        # Default commands
        if component_commands is None:
            component_commands = [["hostname"] for _ in config.components]
        
        if len(component_commands) != len(config.components):
            raise ValueError(
                f"Number of command lists ({len(component_commands)}) "
                f"must match number of components ({len(config.components)})"
            )
        
        # Generate script
        script_path = self.generate_hetjob_script(config, component_commands)
        
        # Submit job
        het_job_id = SlurmCLI.submit_hetjob(script_path, account=config.base_config.account)
        
        # Component IDs are het_job_id, het_job_id+1, het_job_id+2, etc.
        component_ids = [f"{het_job_id}+{i}" for i in range(len(config.components))]
        
        submission = HetJobSubmission(
            het_job_id=het_job_id,
            component_ids=component_ids,
            config=config,
            script_path=script_path,
            submit_time=time.time()
        )
        
        logger.info(f"Submitted hetjob {het_job_id} with components {component_ids}")
        return submission
    
    def wait_for_completion(
        self,
        submission: Union[JobSubmission, HetJobSubmission],
        timeout: int = 3600,
        poll_interval: int = 60,
        verbose: bool = False
    ) -> str:
        """
        Wait for a job to complete.
        
        Args:
            submission: Job submission to wait for
            timeout: Maximum wait time in seconds
            poll_interval: Polling interval (minimum 60s)
            verbose: Print status updates
        
        Returns:
            Final job state
        """
        if isinstance(submission, HetJobSubmission):
            job_id = submission.het_job_id
        else:
            job_id = submission.job_id
        
        return SlurmCLI.wait_for_job(
            job_id,
            timeout=timeout,
            poll_interval=poll_interval,
            verbose=verbose
        )
    
    def cancel_job(self, submission: Union[JobSubmission, HetJobSubmission]) -> bool:
        """
        Cancel a submitted job.
        
        Args:
            submission: Job submission to cancel
        
        Returns:
            True if cancellation succeeded
        """
        if isinstance(submission, HetJobSubmission):
            job_id = submission.het_job_id
        else:
            job_id = submission.job_id
        
        return SlurmCLI.cancel_job(job_id)
