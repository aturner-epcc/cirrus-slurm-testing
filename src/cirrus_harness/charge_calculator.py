"""
Charge calculation for Slurm jobs.

Computes core-hours consumed based on Slurm accounting data.
"""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass
import logging

from cirrus_harness.utils.slurm_commands import SlurmCLI, SacctRecord
from cirrus_harness.utils.validators import get_qos_multiplier, CIRRAS_SPECS
from cirrus_harness.job_submitter import SlurmJobConfig, HetJobConfig

logger = logging.getLogger(__name__)


@dataclass
class JobChargeRecord:
    """
    Record of job resource usage and charge.
    
    Attributes:
        job_id: Slurm job ID
        partition: Partition used
        qos: QoS used
        elapsed_seconds: Actual runtime in seconds
        allocated_cores: Number of cores allocated
        exclusive: Whether exclusive access was used
        effective_cores: Cores charged for (adjusted for exclusive)
        base_core_hours: Core-hours before QoS adjustment
        qos_multiplier: QoS charge multiplier
        charged_core_hours: Final charged core-hours
        state: Final job state
        exit_code: Job exit code
    """
    job_id: str
    partition: str
    qos: str
    elapsed_seconds: int
    allocated_cores: int
    exclusive: bool
    effective_cores: int
    base_core_hours: float
    qos_multiplier: float
    charged_core_hours: float
    state: str
    exit_code: str
    
    def __str__(self) -> str:
        """Human-readable summary."""
        return (
            f"Job {self.job_id}:\n"
            f"  State: {self.state} (Exit: {self.exit_code})\n"
            f"  Elapsed: {self.elapsed_seconds}s ({self.elapsed_seconds/3600:.2f}h)\n"
            f"  Cores: {self.allocated_cores} allocated, {self.effective_cores} charged\n"
            f"  Base core-hours: {self.base_core_hours:.2f}\n"
            f"  QoS multiplier: {self.qos_multiplier}×\n"
            f"  Charged core-hours: {self.charged_core_hours:.2f}"
        )


@dataclass
class ArrayChargeSummary:
    """
    Summary of charges for a job array.
    
    Attributes:
        parent_job_id: Parent array job ID
        num_tasks: Number of array tasks
        individual_charges: List of individual task charges
        total_core_hours: Sum of all charges
    """
    parent_job_id: str
    num_tasks: int
    individual_charges: List[JobChargeRecord]
    total_core_hours: float
    
    def __str__(self) -> str:
        return (
            f"Job Array {self.parent_job_id}:\n"
            f"  Tasks: {self.num_tasks}\n"
            f"  Total charged core-hours: {self.total_core_hours:.2f}"
        )


@dataclass
class HetJobChargeSummary:
    """
    Summary of charges for a heterogeneous job.
    
    Attributes:
        het_job_id: Heterogeneous job leader ID
        component_charges: List of per-component charges
        total_core_hours: Sum of all component charges
    """
    het_job_id: str
    component_charges: List[JobChargeRecord]
    total_core_hours: float
    
    def __str__(self) -> str:
        lines = [f"Heterogeneous Job {self.het_job_id}:"]
        for i, charge in enumerate(self.component_charges):
            lines.append(f"  Component {i}: {charge.charged_core_hours:.2f} core-hours")
        lines.append(f"  Total charged core-hours: {self.total_core_hours:.2f}")
        return "\n".join(lines)


class ChargeCalculator:
    """
    Compute job charges from Slurm accounting data.
    
    Core-hours are calculated as:
        (effective_cores × elapsed_seconds) / 3600
    
    Where effective_cores depends on:
    - With --exclusive: entire node (288 cores on Cirrus)
    - Without --exclusive: actual cores requested
    
    QoS multipliers:
    - standard, long, short, largescale: 1.0×
    - highpriority, reservation: 1.5×
    - lowpriority: 0.0× (free)
    
    Example:
        >>> calculator = ChargeCalculator()
        >>> charge = calculator.compute_charge("12345", config)
        >>> print(f"Charged {charge.charged_core_hours:.2f} core-hours")
    """
    
    # QoS charge multipliers
    QOS_MULTIPLIERS: Dict[str, float] = {
        "standard": 1.0,
        "long": 1.0,
        "short": 1.0,
        "largescale": 1.0,
        "highpriority": 1.5,
        "lowpriority": 0.0,
        "reservation": 1.5,
    }
    
    # Cirrus hardware
    CORES_PER_NODE = CIRRAS_SPECS["cores_per_node"]  # 288
    
    def __init__(self, default_exclusive: bool = False):
        """
        Initialize charge calculator.
        
        Args:
            default_exclusive: Default assumption for exclusive access
        """
        self.default_exclusive = default_exclusive
    
    def compute_charge(
        self,
        job_id: str,
        config: Optional[SlurmJobConfig] = None,
        accounting_record: Optional[SacctRecord] = None
    ) -> JobChargeRecord:
        """
        Compute charge for a single job.
        
        Args:
            job_id: Slurm job ID
            config: Job configuration (for exclusive flag)
            accounting_record: Pre-fetched accounting data (optional)
        
        Returns:
            JobChargeRecord with charge details
        
        Raises:
            ValueError: If job not found or accounting data unavailable
        """
        # Get accounting data
        if accounting_record is None:
            records = SlurmCLI.get_job_accounting(job_id, whole_hetjob=False)
            if not records:
                raise ValueError(f"No accounting data found for job {job_id}")
            accounting_record = records[0]
        
        # Determine if exclusive
        exclusive = self.default_exclusive
        if config:
            exclusive = config.exclusive
        
        # Calculate effective cores
        if exclusive:
            # Charge for entire nodes
            effective_cores = accounting_record.allocated_nodes * self.CORES_PER_NODE
        else:
            # Charge for actual cores allocated
            effective_cores = accounting_record.allocated_cpus
        
        # Calculate core-hours
        elapsed_seconds = accounting_record.elapsed_raw
        base_core_hours = (effective_cores * elapsed_seconds) / 3600.0
        
        # Apply QoS multiplier
        qos = accounting_record.qos or (config.qos if config else "standard")
        qos_multiplier = self.QOS_MULTIPLIERS.get(qos.lower(), 1.0)
        charged_core_hours = base_core_hours * qos_multiplier
        
        return JobChargeRecord(
            job_id=accounting_record.job_id,
            partition=accounting_record.partition,
            qos=qos,
            elapsed_seconds=elapsed_seconds,
            allocated_cores=accounting_record.allocated_cpus,
            exclusive=exclusive,
            effective_cores=effective_cores,
            base_core_hours=base_core_hours,
            qos_multiplier=qos_multiplier,
            charged_core_hours=charged_core_hours,
            state=accounting_record.state,
            exit_code=accounting_record.exit_code,
        )
    
    def compute_array_charge(
        self,
        parent_job_id: str,
        config: Optional[SlurmJobConfig] = None
    ) -> ArrayChargeSummary:
        """
        Compute charges for all elements of a job array.
        
        Args:
            parent_job_id: Parent array job ID
            config: Job configuration (for exclusive flag)
        
        Returns:
            ArrayChargeSummary with per-task and total charges
        """
        # Get all array task records
        records = SlurmCLI.get_job_accounting(parent_job_id, whole_hetjob=False)
        
        if not records:
            raise ValueError(f"No accounting data found for array job {parent_job_id}")
        
        # Filter to array tasks only (those with array_task_id)
        array_records = [r for r in records if r.array_task_id is not None]
        
        if not array_records:
            # No array tasks found, treat as single job
            charge = self.compute_charge(parent_job_id, config, records[0])
            return ArrayChargeSummary(
                parent_job_id=parent_job_id,
                num_tasks=1,
                individual_charges=[charge],
                total_core_hours=charge.charged_core_hours
            )
        
        # Compute charge for each task
        individual_charges = []
        for record in array_records:
            charge = self.compute_charge(record.job_id, config, record)
            individual_charges.append(charge)
        
        # Sum total
        total_core_hours = sum(c.charged_core_hours for c in individual_charges)
        
        return ArrayChargeSummary(
            parent_job_id=parent_job_id,
            num_tasks=len(array_records),
            individual_charges=individual_charges,
            total_core_hours=total_core_hours
        )
    
    def compute_hetjob_charge(
        self,
        het_job_id: str,
        config: Optional[HetJobConfig] = None
    ) -> HetJobChargeSummary:
        """
        Compute charges for all components of a heterogeneous job.
        
        Args:
            het_job_id: Heterogeneous job leader ID
            config: HetJob configuration (for exclusive flags)
        
        Returns:
            HetJobChargeSummary with per-component and total charges
        """
        # Get all component records
        records = SlurmCLI.get_job_accounting(het_job_id, whole_hetjob=True)
        
        if not records:
            raise ValueError(f"No accounting data found for hetjob {het_job_id}")
        
        # Compute charge for each component
        component_charges = []
        for record in records:
            # Determine if this component used exclusive
            exclusive = self.default_exclusive
            if config and config.base_config.exclusive:
                exclusive = True
            
            charge = self.compute_charge(record.job_id, None, record)
            # Override exclusive flag if needed
            charge = JobChargeRecord(
                job_id=charge.job_id,
                partition=charge.partition,
                qos=charge.qos,
                elapsed_seconds=charge.elapsed_seconds,
                allocated_cores=charge.allocated_cores,
                exclusive=exclusive,
                effective_cores=charge.effective_cores,
                base_core_hours=charge.base_core_hours,
                qos_multiplier=charge.qos_multiplier,
                charged_core_hours=charge.charged_core_hours,
                state=charge.state,
                exit_code=charge.exit_code,
            )
            component_charges.append(charge)
        
        # Sum total
        total_core_hours = sum(c.charged_core_hours for c in component_charges)
        
        return HetJobChargeSummary(
            het_job_id=het_job_id,
            component_charges=component_charges,
            total_core_hours=total_core_hours
        )
    
    def compute_total_charge(
        self,
        job_id: str,
        config: Optional[SlurmJobConfig] = None,
        is_array: bool = False,
        is_hetjob: bool = False
    ) -> float:
        """
        Compute total charge for a job (handles arrays and hetjobs).
        
        Args:
            job_id: Job ID
            config: Job configuration
            is_array: Whether this is a job array
            is_hetjob: Whether this is a heterogeneous job
        
        Returns:
            Total charged core-hours
        """
        if is_hetjob:
            het_config = None
            if config:
                # Convert SlurmJobConfig to HetJobConfig if needed
                pass
            summary = self.compute_hetjob_charge(job_id, het_config)
            return summary.total_core_hours
        elif is_array:
            summary = self.compute_array_charge(job_id, config)
            return summary.total_core_hours
        else:
            charge = self.compute_charge(job_id, config)
            return charge.charged_core_hours
    
    def get_qos_multiplier(self, qos: str) -> float:
        """
        Get the charge multiplier for a QoS.
        
        Args:
            qos: QoS name
        
        Returns:
            Charge multiplier
        """
        return self.QOS_MULTIPLIERS.get(qos.lower(), 1.0)
    
    def explain_charge(
        self,
        job_id: str,
        config: Optional[SlurmJobConfig] = None
    ) -> str:
        """
        Generate a detailed explanation of job charges.
        
        Args:
            job_id: Job ID
            config: Job configuration
        
        Returns:
            Human-readable explanation
        """
        try:
            charge = self.compute_charge(job_id, config)
            
            lines = [
                f"Charge Breakdown for Job {job_id}",
                "=" * 50,
                f"Partition: {charge.partition}",
                f"QoS: {charge.qos}",
                f"State: {charge.state}",
                f"Exit Code: {charge.exit_code}",
                "",
                "Resource Usage:",
                f"  Elapsed Time: {charge.elapsed_seconds}s ({charge.elapsed_seconds/60:.1f} minutes)",
                f"  Allocated Cores: {charge.allocated_cores}",
                f"  Exclusive Access: {'Yes' if charge.exclusive else 'No'}",
                f"  Effective Cores: {charge.effective_cores}",
                "",
                "Charge Calculation:",
                f"  Base Core-Hours: ({charge.effective_cores} cores × {charge.elapsed_seconds}s) / 3600 = {charge.base_core_hours:.2f}",
                f"  QoS Multiplier: {charge.qos_multiplier}×",
                f"  Final Charge: {charge.base_core_hours:.2f} × {charge.qos_multiplier} = {charge.charged_core_hours:.2f} core-hours",
            ]
            
            if charge.exclusive and charge.allocated_cores != charge.effective_cores:
                lines.append("")
                lines.append(f"Note: --exclusive was used, charging for full nodes ({charge.effective_cores} cores)")
                lines.append(f"      instead of requested cores ({charge.allocated_cores})")
            
            if charge.qos_multiplier == 0.0:
                lines.append("")
                lines.append("Note: lowpriority QoS - no charge applied")
            elif charge.qos_multiplier > 1.0:
                lines.append("")
                lines.append(f"Note: {charge.qos} QoS incurs {charge.qos_multiplier}× surcharge")
            
            return "\n".join(lines)
            
        except ValueError as e:
            return f"Error computing charge: {e}"
