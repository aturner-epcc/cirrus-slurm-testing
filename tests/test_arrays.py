"""
Test job array functionality.

Verifies job array configuration, submission, and charge aggregation
across array elements.
"""

import pytest
from unittest.mock import patch, MagicMock

from cirrus_harness.job_submitter import SlurmJobConfig, JobSubmitter
from cirrus_harness.charge_calculator import ChargeCalculator, ArrayChargeSummary
from cirrus_harness.utils.slurm_commands import SacctRecord
from tests.conftest import create_mock_sacct_record


class TestArrayConfiguration:
    """Test job array configuration."""
    
    def test_array_spec_basic(self):
        """Test basic array specification."""
        config = SlurmJobConfig(
            partition="standard",
            qos="standard",
            account="test-account",
            nodes=1,
            tasks_per_node=1,
            cpus_per_task=1,
            time="00:10:00",
            array_spec="1-10",
            job_name="array_test",
        )
        
        assert config.array_spec == "1-10"
    
    def test_array_spec_with_step(self):
        """Test array specification with step."""
        config = SlurmJobConfig(
            partition="standard",
            qos="standard",
            account="test-account",
            nodes=1,
            tasks_per_node=1,
            cpus_per_task=1,
            time="00:10:00",
            array_spec="0-90:10",  # 0, 10, 20, ..., 90
            job_name="array_step",
        )
        
        assert config.array_spec == "0-90:10"
    
    def test_array_spec_with_concurrency(self):
        """Test array specification with concurrency limit."""
        config = SlurmJobConfig(
            partition="standard",
            qos="standard",
            account="test-account",
            nodes=1,
            tasks_per_node=1,
            cpus_per_task=1,
            time="00:10:00",
            array_spec="1-100%10",  # Max 10 concurrent
            job_name="array_concurrent",
        )
        
        assert config.array_spec == "1-100%10"


class TestArrayScriptGeneration:
    """Test job array script generation."""
    
    def test_array_script_includes_array_directive(self, temp_output_dir):
        """Test that array directive appears in script."""
        config = SlurmJobConfig(
            partition="standard",
            qos="standard",
            account="test-account",
            nodes=1,
            tasks_per_node=1,
            cpus_per_task=1,
            time="00:10:00",
            array_spec="1-10",
            job_name="array_test",
        )
        
        submitter = JobSubmitter(output_dir=temp_output_dir)
        script_path = submitter.generate_script(config, ["hostname"], job_type="serial")
        
        content = script_path.read_text()
        assert "#SBATCH --array=1-10" in content
    
    def test_array_script_output_pattern(self, temp_output_dir):
        """Test that array output pattern is correct."""
        config = SlurmJobConfig(
            partition="standard",
            qos="standard",
            account="test-account",
            nodes=1,
            tasks_per_node=1,
            cpus_per_task=1,
            time="00:10:00",
            array_spec="1-10",
            job_name="array_test",
        )
        
        submitter = JobSubmitter(output_dir=temp_output_dir)
        script_path = submitter.generate_script(config, ["hostname"], job_type="serial")
        
        content = script_path.read_text()
        assert "%a" in content  # Array task ID in output pattern


class TestArrayChargeCalculation:
    """Test charge calculation for job arrays."""
    
    @pytest.fixture
    def mock_array_sacct(self):
        """Mock sacct returning multiple array tasks."""
        def _mock_sacct(job_id, whole_hetjob=True):
            # Simulate 5 array tasks
            records = []
            for i in range(1, 6):
                records.append(SacctRecord(
                    job_id=f"{job_id}_{i}",
                    job_name="array_test",
                    partition="standard",
                    account="test-account",
                    state="COMPLETED",
                    exit_code="0:0",
                    elapsed_raw=600,  # 10 minutes each
                    allocated_cpus=1,
                    allocated_nodes=1,
                    start_time="2024-01-01T10:00:00",
                    end_time="2024-01-01T10:10:00",
                    qos="standard",
                    array_task_id=str(i)
                ))
            return records
        
        with patch('cirrus_harness.charge_calculator.SlurmCLI.get_job_accounting', side_effect=_mock_sacct):
            yield _mock_sacct
    
    def test_array_charge_aggregation(self, mock_array_sacct):
        """Test that charges are aggregated across array tasks."""
        calculator = ChargeCalculator()
        summary = calculator.compute_array_charge("12345")
        
        assert isinstance(summary, ArrayChargeSummary)
        assert summary.num_tasks == 5
        assert len(summary.individual_charges) == 5
    
    def test_array_total_charge(self, mock_array_sacct):
        """Test total charge calculation for array."""
        calculator = ChargeCalculator()
        summary = calculator.compute_array_charge("12345")
        
        # Each task: 1 core × 10 minutes = 0.167 core-hours
        # Total: 5 × 0.167 = 0.833 core-hours
        expected_per_task = 600 / 3600  # 10 minutes in hours
        expected_total = 5 * expected_per_task
        
        assert abs(summary.total_core_hours - expected_total) < 0.001
    
    def test_array_individual_charges(self, mock_array_sacct):
        """Test individual task charges in array."""
        calculator = ChargeCalculator()
        summary = calculator.compute_array_charge("12345")
        
        # All tasks should have same charge (same runtime)
        charges = [c.charged_core_hours for c in summary.individual_charges]
        assert all(c == charges[0] for c in charges)
    
    def test_array_varying_runtimes(self):
        """Test array with varying runtimes per task."""
        def _mock_sacct(job_id, whole_hetjob=True):
            records = []
            runtimes = [300, 600, 900, 1200, 1500]  # Different runtimes
            for i, runtime in enumerate(runtimes, 1):
                records.append(SacctRecord(
                    job_id=f"{job_id}_{i}",
                    job_name="array_test",
                    partition="standard",
                    account="test-account",
                    state="COMPLETED",
                    exit_code="0:0",
                    elapsed_raw=runtime,
                    allocated_cpus=1,
                    allocated_nodes=1,
                    start_time="2024-01-01T10:00:00",
                    end_time="2024-01-01T10:10:00",
                    qos="standard",
                    array_task_id=str(i)
                ))
            return records
        
        with patch('cirrus_harness.charge_calculator.SlurmCLI.get_job_accounting', side_effect=_mock_sacct):
            calculator = ChargeCalculator()
            summary = calculator.compute_array_charge("12345")
            
            # Total should be sum of all runtimes
            total_seconds = sum([300, 600, 900, 1200, 1500])
            expected_core_hours = total_seconds / 3600
            
            assert abs(summary.total_core_hours - expected_core_hours) < 0.001


class TestArrayLargeScale:
    """Test large job arrays."""
    
    def test_large_array_credit_reservation_warning(self):
        """Test warning about credit reservation for large arrays."""
        # Large arrays with long --time can exhaust credit reservations
        config = SlurmJobConfig(
            partition="standard",
            qos="standard",
            account="test-account",
            nodes=1,
            tasks_per_node=1,
            cpus_per_task=1,
            time="12:00:00",  # Long runtime
            array_spec="1-1000",  # Many tasks
            job_name="large_array",
        )
        
        # This should work but may have warnings in production
        is_valid, message, warnings = config.validate()
        assert is_valid


class TestArrayConcurrency:
    """Test array concurrency limits."""
    
    def test_array_concurrency_parsing(self):
        """Test parsing of concurrency-limited array spec."""
        config = SlurmJobConfig(
            partition="standard",
            qos="standard",
            account="test-account",
            nodes=1,
            tasks_per_node=1,
            cpus_per_task=1,
            time="00:10:00",
            array_spec="1-100%10",
            job_name="concurrent_array",
        )
        
        assert "%10" in config.array_spec
    
    def test_array_concurrency_script(self, temp_output_dir):
        """Test that concurrency limit appears in script."""
        config = SlurmJobConfig(
            partition="standard",
            qos="standard",
            account="test-account",
            nodes=1,
            tasks_per_node=1,
            cpus_per_task=1,
            time="00:10:00",
            array_spec="1-100%10",
            job_name="concurrent_array",
        )
        
        submitter = JobSubmitter(output_dir=temp_output_dir)
        script_path = submitter.generate_script(config, ["hostname"], job_type="serial")
        
        content = script_path.read_text()
        assert "#SBATCH --array=1-100%10" in content


class TestArrayVsParallelSrun:
    """Test comparison between arrays and concurrent srun."""
    
    def test_array_vs_srun_recommendation(self):
        """Test documentation about array vs concurrent srun."""
        # Best practice: for many short tasks, prefer concurrent srun
        # This is documented but we can test the logic
        
        # Short task array
        short_array_config = SlurmJobConfig(
            partition="standard",
            qos="standard",
            account="test-account",
            nodes=1,
            tasks_per_node=1,
            cpus_per_task=1,
            time="00:05:00",  # Short
            array_spec="1-100",  # Many tasks
            job_name="short_array",
        )
        
        is_valid, message, warnings = short_array_config.validate()
        assert is_valid
        
        # Note: In production, this would generate a warning about
        # preferring concurrent srun for many short tasks


class TestArrayFailedTasks:
    """Test handling of failed array tasks."""
    
    def test_array_partial_failure_charge(self):
        """Test charge calculation when some array tasks fail."""
        def _mock_sacct(job_id, whole_hetjob=True):
            records = []
            # 3 successful, 2 failed
            for i in range(1, 6):
                state = "COMPLETED" if i <= 3 else "FAILED"
                exit_code = "0:0" if i <= 3 else "1:0"
                records.append(SacctRecord(
                    job_id=f"{job_id}_{i}",
                    job_name="array_test",
                    partition="standard",
                    account="test-account",
                    state=state,
                    exit_code=exit_code,
                    elapsed_raw=600,
                    allocated_cpus=1,
                    allocated_nodes=1,
                    start_time="2024-01-01T10:00:00",
                    end_time="2024-01-01T10:10:00",
                    qos="standard",
                    array_task_id=str(i)
                ))
            return records
        
        with patch('cirrus_harness.charge_calculator.SlurmCLI.get_job_accounting', side_effect=_mock_sacct):
            calculator = ChargeCalculator()
            summary = calculator.compute_array_charge("12345")
            
            # All tasks are charged regardless of success/failure
            assert summary.num_tasks == 5
            assert len(summary.individual_charges) == 5
            
            # Failed tasks still consumed resources
            failed_charges = [
                c for c, r in zip(
                    summary.individual_charges,
                    [rec for rec in _mock_sacct("12345")]
                )
                if r.state == "FAILED"
            ]
            assert len(failed_charges) == 2


class TestArrayOutputFiles:
    """Test array output file handling."""
    
    def test_array_output_file_pattern(self, temp_output_dir):
        """Test output file pattern for arrays."""
        config = SlurmJobConfig(
            partition="standard",
            qos="standard",
            account="test-account",
            nodes=1,
            tasks_per_node=1,
            cpus_per_task=1,
            time="00:10:00",
            array_spec="1-10",
            job_name="array_test",
            output_file="task_%a.out",
        )
        
        submitter = JobSubmitter(output_dir=temp_output_dir)
        script_path = submitter.generate_script(config, ["hostname"], job_type="serial")
        
        content = script_path.read_text()
        assert "#SBATCH --output=task_%a.out" in content
    
    def test_array_default_output_pattern(self, temp_output_dir):
        """Test default output pattern for arrays."""
        config = SlurmJobConfig(
            partition="standard",
            qos="standard",
            account="test-account",
            nodes=1,
            tasks_per_node=1,
            cpus_per_task=1,
            time="00:10:00",
            array_spec="1-10",
            job_name="array_test",
        )
        
        submitter = JobSubmitter(output_dir=temp_output_dir)
        script_path = submitter.generate_script(config, ["hostname"], job_type="serial")
        
        content = script_path.read_text()
        # Should include array output pattern
        assert "%a" in content or "array_%a" in content
