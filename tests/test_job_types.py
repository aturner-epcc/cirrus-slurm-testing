"""
Test different job types: serial, MPI, and MPI+OpenMP.

Verifies job configuration, script generation, and charge calculations
for each job type.
"""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from cirrus_harness.job_submitter import SlurmJobConfig, JobSubmitter
from cirrus_harness.charge_calculator import ChargeCalculator
from cirrus_harness.utils.validators import CIRRAS_SPECS
from tests.conftest import create_mock_sacct_record


class TestSerialJobs:
    """Test serial (single-core) job functionality."""
    
    def test_serial_config_creation(self):
        """Test serial job configuration."""
        config = SlurmJobConfig(
            partition="standard",
            qos="standard",
            account="test-account",
            nodes=1,
            tasks_per_node=1,
            cpus_per_task=1,
            time="00:10:00",
            job_name="serial_test",
        )
        
        assert config.total_cores == 1
        assert config.effective_cores == 1
    
    def test_serial_script_generation(self, temp_output_dir):
        """Test serial job script generation."""
        config = SlurmJobConfig(
            partition="standard",
            qos="standard",
            account="test-account",
            nodes=1,
            tasks_per_node=1,
            cpus_per_task=1,
            time="00:10:00",
            job_name="serial_test",
        )
        
        submitter = JobSubmitter(output_dir=temp_output_dir)
        script_path = submitter.generate_script(
            config,
            commands=["echo 'Hello from serial job'"],
            job_type="serial"
        )
        
        assert script_path.exists()
        
        # Verify script content
        content = script_path.read_text()
        assert "#SBATCH --ntasks=1" in content
        assert "export OMP_NUM_THREADS=1" in content
        assert "srun" in content
    
    def test_serial_charge_calculation(self):
        """Test charge calculation for serial jobs."""
        with patch('cirrus_harness.charge_calculator.SlurmCLI.get_job_accounting') as mock_sacct:
            mock_sacct.return_value = [create_mock_sacct_record(
                elapsed_seconds=3600,  # 1 hour
                allocated_cpus=1,
                allocated_nodes=1,
                qos="standard"
            )]
            
            calculator = ChargeCalculator()
            charge = calculator.compute_charge("12345")
            
            assert charge.allocated_cores == 1
            assert charge.base_core_hours == 1.0  # 1 core × 1 hour
            assert charge.charged_core_hours == 1.0
    
    def test_serial_long_runtime(self):
        """Test serial job with long runtime."""
        with patch('cirrus_harness.charge_calculator.SlurmCLI.get_job_accounting') as mock_sacct:
            # 24 hour serial job
            mock_sacct.return_value = [create_mock_sacct_record(
                elapsed_seconds=86400,  # 24 hours
                allocated_cpus=1,
                qos="standard"
            )]
            
            calculator = ChargeCalculator()
            charge = calculator.compute_charge("12345")
            
            assert charge.base_core_hours == 24.0  # 1 core × 24 hours


class TestMPIJobs:
    """Test MPI (pure distributed memory) job functionality."""
    
    def test_mpi_single_node_config(self):
        """Test single-node MPI job configuration."""
        config = SlurmJobConfig(
            partition="standard",
            qos="standard",
            account="test-account",
            nodes=1,
            tasks_per_node=288,
            cpus_per_task=1,
            time="00:20:00",
            exclusive=True,
            job_name="mpi_test",
        )
        
        assert config.total_cores == 288
        assert config.effective_cores == 288  # Same with exclusive
    
    def test_mpi_multi_node_config(self):
        """Test multi-node MPI job configuration."""
        config = SlurmJobConfig(
            partition="standard",
            qos="standard",
            account="test-account",
            nodes=4,
            tasks_per_node=288,
            cpus_per_task=1,
            time="01:00:00",
            exclusive=True,
            job_name="mpi_multinode",
        )
        
        assert config.total_cores == 1152  # 4 × 288
        assert config.effective_cores == 1152
    
    def test_mpi_script_generation(self, temp_output_dir):
        """Test MPI job script generation."""
        config = SlurmJobConfig(
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
        
        submitter = JobSubmitter(output_dir=temp_output_dir)
        script_path = submitter.generate_script(
            config,
            commands=["./mpi_app.x"],
            job_type="mpi"
        )
        
        assert script_path.exists()
        
        content = script_path.read_text()
        assert "#SBATCH --nodes=2" in content
        assert "#SBATCH --tasks-per-node=288" in content
        assert "#SBATCH --exclusive" in content
        assert "--hint=nomultithread" in content
        assert "--distribution=block:block" in content
    
    def test_mpi_charge_calculation(self):
        """Test charge calculation for MPI jobs."""
        with patch('cirrus_harness.charge_calculator.SlurmCLI.get_job_accounting') as mock_sacct:
            mock_sacct.return_value = [create_mock_sacct_record(
                elapsed_seconds=3600,  # 1 hour
                allocated_cpus=576,  # 2 nodes × 288 cores
                allocated_nodes=2,
                qos="standard"
            )]
            
            calculator = ChargeCalculator()
            charge = calculator.compute_charge("12345")
            
            assert charge.allocated_cores == 576
            assert charge.base_core_hours == 576.0  # 576 cores × 1 hour
    
    def test_mpi_partial_node_no_exclusive(self):
        """Test MPI job using partial node without exclusive."""
        config = SlurmJobConfig(
            partition="standard",
            qos="standard",
            account="test-account",
            nodes=1,
            tasks_per_node=64,  # Partial node
            cpus_per_task=1,
            time="00:10:00",
            exclusive=False,
            job_name="mpi_partial",
        )
        
        assert config.total_cores == 64
        assert config.effective_cores == 64  # No exclusive adjustment
    
    def test_mpi_multi_node_requires_exclusive(self):
        """Test that multi-node MPI jobs warn about exclusive."""
        config = SlurmJobConfig(
            partition="standard",
            qos="standard",
            account="test-account",
            nodes=2,
            tasks_per_node=288,
            cpus_per_task=1,
            time="00:20:00",
            exclusive=False,  # Should warn
        )
        
        is_valid, message, warnings = config.validate()
        assert any("exclusive" in w.lower() for w in warnings)


class TestHybridJobs:
    """Test hybrid MPI+OpenMP job functionality."""
    
    def test_hybrid_config_24x12(self):
        """Test hybrid job with 24 MPI ranks × 12 OpenMP threads."""
        config = SlurmJobConfig(
            partition="standard",
            qos="standard",
            account="test-account",
            nodes=1,
            tasks_per_node=24,
            cpus_per_task=12,
            time="00:30:00",
            exclusive=True,
            job_name="hybrid_24x12",
        )
        
        assert config.total_cores == 288  # 24 × 12
        assert config.effective_cores == 288
    
    def test_hybrid_config_36x8(self):
        """Test hybrid job with 36 MPI ranks × 8 OpenMP threads."""
        config = SlurmJobConfig(
            partition="standard",
            qos="standard",
            account="test-account",
            nodes=1,
            tasks_per_node=36,
            cpus_per_task=8,
            time="00:30:00",
            exclusive=True,
            job_name="hybrid_36x8",
        )
        
        assert config.total_cores == 288  # 36 × 8
    
    def test_hybrid_script_generation(self, temp_output_dir):
        """Test hybrid job script generation."""
        config = SlurmJobConfig(
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
        
        submitter = JobSubmitter(output_dir=temp_output_dir)
        script_path = submitter.generate_script(
            config,
            commands=["./hybrid_app.x"],
            job_type="hybrid"
        )
        
        assert script_path.exists()
        
        content = script_path.read_text()
        assert "export OMP_NUM_THREADS=" in content
        assert "export OMP_PLACES=cores" in content
        assert "export SRUN_CPUS_PER_TASK=" in content
    
    def test_hybrid_charge_calculation(self):
        """Test charge calculation for hybrid jobs."""
        with patch('cirrus_harness.charge_calculator.SlurmCLI.get_job_accounting') as mock_sacct:
            mock_sacct.return_value = [create_mock_sacct_record(
                elapsed_seconds=3600,  # 1 hour
                allocated_cpus=288,
                allocated_nodes=1,
                qos="standard"
            )]
            
            calculator = ChargeCalculator()
            charge = calculator.compute_charge("12345")
            
            assert charge.allocated_cores == 288
            assert charge.base_core_hours == 288.0
    
    def test_hybrid_multi_node(self):
        """Test multi-node hybrid job."""
        config = SlurmJobConfig(
            partition="standard",
            qos="standard",
            account="test-account",
            nodes=4,
            tasks_per_node=24,
            cpus_per_task=12,
            time="02:00:00",
            exclusive=True,
            job_name="hybrid_multinode",
        )
        
        assert config.total_cores == 1152  # 4 × 288
        assert config.effective_cores == 1152
        
        with patch('cirrus_harness.charge_calculator.SlurmCLI.get_job_accounting') as mock_sacct:
            mock_sacct.return_value = [create_mock_sacct_record(
                elapsed_seconds=7200,  # 2 hours
                allocated_cpus=1152,
                allocated_nodes=4,
                qos="standard"
            )]
            
            calculator = ChargeCalculator()
            charge = calculator.compute_charge("12345")
            
            assert charge.base_core_hours == 2304.0  # 1152 cores × 2 hours


class TestOpenMPJobs:
    """Test OpenMP-only (shared memory) job functionality."""
    
    def test_openmp_full_node_config(self):
        """Test OpenMP job using full node."""
        config = SlurmJobConfig(
            partition="standard",
            qos="standard",
            account="test-account",
            nodes=1,
            tasks_per_node=1,
            cpus_per_task=288,
            time="00:30:00",
            job_name="openmp_full",
        )
        
        assert config.total_cores == 288
    
    def test_openmp_partial_node_config(self):
        """Test OpenMP job using partial node."""
        config = SlurmJobConfig(
            partition="standard",
            qos="standard",
            account="test-account",
            nodes=1,
            tasks_per_node=1,
            cpus_per_task=64,
            time="00:30:00",
            job_name="openmp_partial",
        )
        
        assert config.total_cores == 64
    
    def test_openmp_script_generation(self, temp_output_dir):
        """Test OpenMP job script generation."""
        config = SlurmJobConfig(
            partition="standard",
            qos="standard",
            account="test-account",
            nodes=1,
            tasks_per_node=1,
            cpus_per_task=128,
            time="00:30:00",
            job_name="openmp_test",
        )
        
        submitter = JobSubmitter(output_dir=temp_output_dir)
        script_path = submitter.generate_script(
            config,
            commands=["./openmp_app.x"],
            job_type="openmp"
        )
        
        assert script_path.exists()
        
        content = script_path.read_text()
        assert "export OMP_NUM_THREADS=" in content
        assert "export OMP_PLACES=cores" in content
    
    def test_openmp_charge_calculation(self):
        """Test charge calculation for OpenMP jobs."""
        with patch('cirrus_harness.charge_calculator.SlurmCLI.get_job_accounting') as mock_sacct:
            mock_sacct.return_value = [create_mock_sacct_record(
                elapsed_seconds=1800,  # 30 minutes
                allocated_cpus=128,
                allocated_nodes=1,
                qos="standard"
            )]
            
            calculator = ChargeCalculator()
            charge = calculator.compute_charge("12345")
            
            assert charge.base_core_hours == 64.0  # 128 cores × 0.5 hours


class TestJobTypeComparison:
    """Compare charges across different job types."""
    
    def test_same_cores_same_charge(self):
        """Test that same core count gives same charge regardless of job type."""
        # All three configurations use 288 cores for 1 hour
        configs = [
            create_mock_sacct_record(allocated_cpus=288, elapsed_seconds=3600),
            create_mock_sacct_record(allocated_cpus=288, elapsed_seconds=3600),
            create_mock_sacct_record(allocated_cpus=288, elapsed_seconds=3600),
        ]
        
        with patch('cirrus_harness.charge_calculator.SlurmCLI.get_job_accounting') as mock_sacct:
            mock_sacct.side_effect = [[c] for c in configs]
            
            calculator = ChargeCalculator()
            
            charge1 = calculator.compute_charge("12345")
            charge2 = calculator.compute_charge("12346")
            charge3 = calculator.compute_charge("12347")
            
            # All should have same charge
            assert charge1.base_core_hours == charge2.base_core_hours == charge3.base_core_hours
            assert charge1.base_core_hours == 288.0
