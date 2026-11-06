"""
Slurm command-line interface wrappers.

Provides safe, validated wrappers for sbatch, squeue, sacct, and scancel.
"""

import subprocess
import re
import time
from typing import Optional, Dict, List, Any
from dataclasses import dataclass
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class SacctRecord:
    """Record from sacct accounting data."""
    job_id: str
    job_name: str
    partition: str
    account: str
    state: str
    exit_code: str
    elapsed_raw: int  # Seconds
    allocated_cpus: int
    allocated_nodes: int
    start_time: str
    end_time: str
    qos: str = ""
    array_task_id: Optional[str] = None
    
    @property
    def elapsed_hours(self) -> float:
        """Elapsed time in hours."""
        return self.elapsed_raw / 3600.0


@dataclass
class SqueueRecord:
    """Record from squeue status query."""
    job_id: str
    partition: str
    name: str
    user: str
    state: str
    time: str
    nodes: int
    nodelist: str


class SlurmCLI:
    """
    Safe wrappers for Slurm commands.
    
    Enforces minimum polling intervals to avoid disrupting the scheduler.
    """
    
    MIN_POLL_INTERVAL = 60  # Seconds - Cirrus policy requirement
    
    @staticmethod
    def submit_job(script_path: Path, account: Optional[str] = None) -> str:
        """
        Submit a job using sbatch.
        
        Args:
            script_path: Path to the job script
            account: Account/budget code (if not in script)
        
        Returns:
            Job ID as string
        
        Raises:
            RuntimeError: If submission fails
        """
        cmd = ["sbatch", "--parsable"]
        
        if account:
            cmd.extend(["--account", account])
        
        cmd.append(str(script_path))
        
        logger.info(f"Submitting job: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=30
            )
            
            job_id = result.stdout.strip()
            logger.info(f"Submitted job {job_id}")
            return job_id
            
        except subprocess.CalledProcessError as e:
            error_msg = e.stderr.strip() if e.stderr else "Unknown sbatch error"
            logger.error(f"Job submission failed: {error_msg}")
            raise RuntimeError(f"sbatch failed: {error_msg}")
        except subprocess.TimeoutExpired:
            raise RuntimeError("sbatch command timed out")
    
    @staticmethod
    def submit_hetjob(script_path: Path, account: Optional[str] = None) -> str:
        """
        Submit a heterogeneous job.
        
        Args:
            script_path: Path to the hetjob script
            account: Account/budget code
        
        Returns:
            Heterogeneous job leader ID
        """
        # Hetjobs are submitted the same way, but script contains hetjob directives
        return SlurmCLI.submit_job(script_path, account)
    
    @staticmethod
    def get_job_status(job_id: str) -> Optional[SqueueRecord]:
        """
        Get current job status from squeue.
        
        Args:
            job_id: Job ID to query
        
        Returns:
            SqueueRecord if job found, None otherwise
        """
        cmd = [
            "squeue",
            "--job", job_id,
            "--Format=JobID,Partition,Name,User,StateCompact,TimeUsed,Nodes,NodeList",
            "--noheader"
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=30
            )
            
            if not result.stdout.strip():
                return None
            
            parts = result.stdout.strip().split()
            if len(parts) < 8:
                return None
            
            return SqueueRecord(
                job_id=parts[0],
                partition=parts[1],
                name=parts[2],
                user=parts[3],
                state=parts[4],
                time=parts[5],
                nodes=int(parts[6]),
                nodelist=parts[7] if len(parts) > 7 else ""
            )
            
        except subprocess.CalledProcessError:
            return None
        except (ValueError, IndexError) as e:
            logger.warning(f"Failed to parse squeue output: {e}")
            return None
    
    @staticmethod
    def get_job_accounting(job_id: str, whole_hetjob: bool = True) -> List[SacctRecord]:
        """
        Get job accounting data from sacct.
        
        Args:
            job_id: Job ID or heterogeneous job leader ID
            whole_hetjob: If True, get all components of hetjob
        
        Returns:
            List of SacctRecord (multiple for arrays/hetjobs)
        """
        cmd = [
            "sacct",
            "--jobs", job_id,
            "--format=JobID,JobName,Partition,Account,State,ExitCode,ElapsedRaw,AllocCPUS,AllocNodes,Start,End,QOS",
            "--noheader",
            "--parsable2"
        ]
        
        if whole_hetjob:
            cmd.append("--whole-hetjob=yes")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=30
            )
            
            records = []
            for line in result.stdout.strip().split('\n'):
                if not line.strip():
                    continue
                
                parts = line.split('|')
                if len(parts) < 12:
                    continue
                
                # Parse array task ID from JobID if present
                job_id_str = parts[0]
                array_task_id = None
                if '_' in job_id_str:
                    job_id_str, array_task_id = job_id_str.split('_', 1)
                
                try:
                    elapsed_raw = int(parts[6]) if parts[6] else 0
                    alloc_cpus = int(parts[7]) if parts[7] else 0
                    alloc_nodes = int(parts[8]) if parts[8] else 0
                except ValueError:
                    logger.warning(f"Failed to parse numeric fields: {line}")
                    continue
                
                records.append(SacctRecord(
                    job_id=parts[0],
                    job_name=parts[1],
                    partition=parts[2],
                    account=parts[3],
                    state=parts[4],
                    exit_code=parts[5],
                    elapsed_raw=elapsed_raw,
                    allocated_cpus=alloc_cpus,
                    allocated_nodes=alloc_nodes,
                    start_time=parts[9],
                    end_time=parts[10],
                    qos=parts[11],
                    array_task_id=array_task_id
                ))
            
            return records
            
        except subprocess.CalledProcessError as e:
            logger.error(f"sacct failed for job {job_id}: {e.stderr}")
            return []
        except Exception as e:
            logger.error(f"Failed to parse sacct output: {e}")
            return []
    
    @staticmethod
    def cancel_job(job_id: str) -> bool:
        """
        Cancel a job using scancel.
        
        Args:
            job_id: Job ID to cancel (can include array task ID)
        
        Returns:
            True if cancellation succeeded
        """
        cmd = ["scancel", job_id]
        
        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=30)
            logger.info(f"Cancelled job {job_id}")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to cancel job {job_id}: {e.stderr}")
            return False
    
    @staticmethod
    def wait_for_job(
        job_id: str,
        timeout: int = 3600,
        poll_interval: int = 60,
        verbose: bool = False
    ) -> str:
        """
        Wait for a job to complete.
        
        Args:
            job_id: Job ID to monitor
            timeout: Maximum wait time in seconds
            poll_interval: Polling interval in seconds (minimum 60)
            verbose: Print status updates
        
        Returns:
            Final job state
        """
        # Enforce minimum poll interval
        poll_interval = max(poll_interval, SlurmCLI.MIN_POLL_INTERVAL)
        
        start_time = time.time()
        last_state = "PENDING"
        
        while time.time() - start_time < timeout:
            status = SlurmCLI.get_job_status(job_id)
            
            if status is None:
                # Job no longer in queue - check if completed
                accounting = SlurmCLI.get_job_accounting(job_id)
                if accounting:
                    final_state = accounting[0].state
                    if verbose:
                        print(f"Job {job_id} completed with state: {final_state}")
                    return final_state
                return "UNKNOWN"
            
            current_state = status.state
            
            if current_state != last_state and verbose:
                print(f"Job {job_id}: {last_state} -> {current_state}")
            
            last_state = current_state
            
            # Check if job is done
            if current_state in ["COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY"]:
                if verbose:
                    print(f"Job {job_id} finished: {current_state}")
                return current_state
            
            time.sleep(poll_interval)
        
        logger.warning(f"Timeout waiting for job {job_id}")
        return "TIMEOUT"
    
    @staticmethod
    def check_account_valid(account: str) -> bool:
        """
        Check if an account/budget code is valid.
        
        Args:
            account: Account code to check
        
        Returns:
            True if account is valid
        """
        cmd = [
            "sacctmgr",
            "show",
            "assoc",
            f"user={Path.home().name}",  # Current user
            "format=account",
            "--noheader"
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=30
            )
            
            accounts = [line.strip() for line in result.stdout.strip().split('\n')]
            return account in accounts
            
        except subprocess.CalledProcessError:
            logger.warning("Could not check account validity")
            return False
    
    @staticmethod
    def get_user_accounts() -> List[str]:
        """
        Get list of accounts available to current user.
        
        Returns:
            List of account codes
        """
        cmd = [
            "sacctmgr",
            "show",
            "assoc",
            f"user={Path.home().name}",
            "format=account",
            "--noheader"
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=30
            )
            
            return [line.strip() for line in result.stdout.strip().split('\n') if line.strip()]
            
        except subprocess.CalledProcessError:
            logger.error("Failed to get user accounts")
            return []
