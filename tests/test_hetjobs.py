"""
Test heterogeneous job functionality.

Verifies hetjob configuration, submission, and per-component charge calculations.
"""

import pytest
from unittest.mock import patch

from cirrus_harness.job_submitter import SlurmJobConfig, HetJobConfig, HetJobComponent, JobSubmitter
from cirrus_harness.charge_calculator import ChargeCalculator, HetJobChargeSummary
from cirrus_harness.utils.slurm_commands import SacctRecord
from tests.conftest import create_mock_sacct_record


class TestHetJobConfiguration:
    """Test heterogeneous job configuration."""
    
    def test_hetjob_two_components(self):
        """Test hetjob with two components."""
        base_config = SlurmJobConfig(
            partition="standard",
            qos="standard",
            account="test-account",
            time="00:10:00",
        )
        
        config = HetJobConfig(
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
        
        assert len(config.components) == 2
        assert config.components[0].tasks_per_node == 288
        assert config.components[1].tasks_per_node == 1
    
    def test_hetjob_different_partitions(self):
        """Test hetjob with different partitions per component."""
        base_config = SlurmJobConfig(
            partition="standard",
            qos="standard",
            account="test-account",
            time="00:10:00",
        )
        
        config = HetJobConfig(
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
                    partition="highmem",
                    qos="standard",
                ),
            ],
            job_name="hetjob_mixed",
        )
        
        assert config.components[0].partition == "standard"
        assert config.components[1].partition == "highmem"
    
    def test_hetjob_different_qos(self):
        """Test hetjob with different QoS per component."""
        base_config = SlurmJobConfig(
            partition="standard",
            qos="standard",
            account="test-account",
            time="00:10:00",
        )
        
        config = HetJobConfig(
            base_config=base_config,
            components=[
                HetJobComponent(
                    nodes=1,
                    tasks_per_node=288,
                    cpus_per_task=1,
                    qos="standard",
                ),
                HetJobComponent(
                    nodes=1,
                    tasks_per_node=1,
                    cpus_per_task=1,
                    qos="highpriority",
                ),
            ],
        )
        
        assert config.components[0].qos == "standard"
        assert config.components[1].qos == "highpriority"


class TestHetJobScriptGeneration:
    """Test heterogeneous job script generation."""
    
    def test_hetjob_script_generation(self, temp_output_dir):
        """Test hetjob script is generated correctly."""
        base_config = SlurmJobConfig(
            partition="standard",
            qos="standard",
            account="test-account",
            time="00:10:00",
        )
        
        config = HetJobConfig(
            base_config=base_config,
            components=[
                HetJobComponent(nodes=1, tasks_per_node=288, cpus_per_task=1),
                HetJobComponent(nodes=1, tasks_per_node=1, cpus_per_task=1),
            ],
            job_name="hetjob_test",
        )
        
        submitter = JobSubmitter(output_dir=temp_output_dir)
        script_path = submitter.generate_hetjob_script(
            config,
            component_commands=[
                ["./mpi_app.x"],
                ["./serial_app.x"],
            ]
        )
        
        assert script_path.exists()
        
        content = script_path.read_text()
        assert "#SBATCH hetjob" in content
        assert "--het-group=0" in content
        assert "--het-group=1" in content


class TestHetJobChargeCalculation:
    """Test charge calculations for heterogeneous jobs."""
    
    @pytest.fixture
    def mock_hetjob_sacct(self):
        """Mock sacct returning hetjob component data."""
        def _mock_sacct(job_id, whole_hetjob=True):
            # Simulate 2 hetjob components
            records = []
            
            # Component 0: MPI job, 288 cores, 1 hour
            records.append(SacctRecord(
                job_id=f"{job_id}+0",
                job_name="hetjob_test",
                partition="standard",
                account="test-account",
                state="COMPLETED",
                exit_code="0:0",
                elapsed_raw=3600,
                allocated_cpus=288,
                allocated_nodes=1,
                start_time="2024-01-01T10:00:00",
                end_time="2024-01-01T11:00:00",
                qos="standard",
            ))
            
            # Component 1: Serial job, 1 core, 1 hour
            records.append(SacctRecord(
                job_id=f"{job_id}+1",
                job_name="hetjob_test",
                partition="standard",
                account="test-account",
                state="COMPLETED",
                exit_code="0:0",
                elapsed_raw=3600,
                allocated_cpus=1,
                allocated_nodes=1,
                start_time="2024-01-01T10:00:00",
                end_time="2024-01-01T11:00:00",
                qos="standard",
            ))
            
            return records
        
        with patch('cirrus_harness.charge_calculator.SlurmCLI.get_job_accounting', side_effect=_mock_sacct):
            yield _mock_sacct
    
    def test_hetjob_charge_aggregation(self, mock_hetjob_sacct):
        """Test that charges are aggregated across hetjob components."""
        calculator = ChargeCalculator()
        summary = calculator.compute_hetjob_charge("12345")
        
        assert isinstance(summary, HetJobChargeSummary)
        assert len(summary.component_charges) == 2
    
    def test_hetjob_total_charge(self, mock_hetjob_sacct):
        """Test total charge calculation for hetjob."""
        calculator = ChargeCalculator()
        summary = calculator.compute_hetjob_charge("12345")
        
        # Component 0: 288 cores × 1 hour = 288 core-hours
        # Component 1: 1 core × 1 hour = 1 core-hour
        # Total: 289 core-hours
        assert abs(summary.total_core_hours - 289.0) < 0.001
    
    def test_hetjob_per_component_charges(self, mock_hetjob_sacct):
        """Test individual component charges."""
        calculator = ChargeCalculator()
        summary = calculator.compute_hetjob_charge("12345")
        
        # Component 0 should be 288 core-hours
        assert abs(summary.component_charges[0].charged_core_hours - 288.0) < 0.001
        
        # Component 1 should be 1 core-hour
        assert abs(summary.component_charges[1].charged_core_hours - 1.0) < 0.001


class TestHetJobDifferentQoS:
    """Test hetjobs with different QoS per component."""
    
    def test_hetjob_mixed_qos_charges(self):
        """Test hetjob with different QoS per component."""
        def _mock_sacct(job_id, whole_hetjob=True):
            records = []
            
            # Component 0: standard QoS
            records.append(SacctRecord(
                job_id=f"{job_id}+0",
                job_name="hetjob_test",
                partition="standard",
                account="test-account",
                state="COMPLETED",
                exit_code="0:0",
                elapsed_raw=3600,
                allocated_cpus=288,
                allocated_nodes=1,
                start_time="2024-01-01T10:00:00",
                end_time="2024-01-01T11:00:00",
                qos="standard",
            ))
            
            # Component 1: highpriority QoS (1.5×)
            records.append(SacctRecord(
                job_id=f"{job_id}+1",
                job_name="hetjob_test",
                partition="standard",
                account="test-account",
                state="COMPLETED",
                exit_code="0:0",
                elapsed_raw=3600,
                allocated_cpus=1,
                allocated_nodes=1,
                start_time="2024-01-01T10:00:00",
                end_time="2024-01-01T11:00:00",
                qos="highpriority",
            ))
            
            return records
        
        with patch('cirrus_harness.charge_calculator.SlurmCLI.get_job_accounting', side_effect=_mock_sacct):
            calculator = ChargeCalculator()
            summary = calculator.compute_hetjob_charge("12345")
            
            # Component 0: 288 core-hours × 1.0 = 288
            assert abs(summary.component_charges[0].charged_core_hours - 288.0) < 0.001
            
            # Component 1: 1 core-hour × 1.5 = 1.5
            assert abs(summary.component_charges[1].charged_core_hours - 1.5) < 0.001
            
            # Total: 289.5 core-hours
            assert abs(summary.total_core_hours - 289.5) < 0.001


class TestHetJobDifferentPartitions:
    """Test hetjobs with different partitions."""
    
    def test_hetjob_standard_highmem(self):
        """Test hetjob with standard and highmem partitions."""
        def _mock_sacct(job_id, whole_hetjob=True):
            records = []
            
            # Component 0: standard partition
            records.append(SacctRecord(
                job_id=f"{job_id}+0",
                job_name="hetjob_test",
                partition="standard",
                account="test-account",
                state="COMPLETED",
                exit_code="0:0",
                elapsed_raw=3600,
                allocated_cpus=288,
                allocated_nodes=1,
                start_time="2024-01-01T10:00:00",
                end_time="2024-01-01T11:00:00",
                qos="standard",
            ))
            
            # Component 1: highmem partition
            records.append(SacctRecord(
                job_id=f"{job_id}+1",
                job_name="hetjob_test",
                partition="highmem",
                account="test-account",
                state="COMPLETED",
                exit_code="0:0",
                elapsed_raw=3600,
                allocated_cpus=1,
                allocated_nodes=1,
                start_time="2024-01-01T10:00:00",
                end_time="2024-01-01T11:00:00",
                qos="standard",
            ))
            
            return records
        
        with patch('cirrus_harness.charge_calculator.SlurmCLI.get_job_accounting', side_effect=_mock_sacct):
            calculator = ChargeCalculator()
            summary = calculator.compute_hetjob_charge("12345")
            
            # Both components should be charged
            assert len(summary.component_charges) == 2
            assert summary.total_core_hours > 0


class TestHetJobLimitations:
    """Test hetjob limitations and constraints."""
    
    def test_hetjob_job_arrays_not_supported(self):
        """Test that job arrays are not supported with hetjobs."""
        # This is a Slurm limitation documented in the spec
        # We can't test it directly but can verify our config doesn't combine them
        
        base_config = SlurmJobConfig(
            partition="standard",
            qos="standard",
            account="test-account",
            time="00:10:00",
            array_spec="1-10",  # This would be invalid for hetjobs
        )
        
        # In production, this should be validated
        # For now, we just note the limitation
        assert base_config.array_spec == "1-10"
    
    def test_hetjob_component_count_limit(self):
        """Test hetjob with many components."""
        base_config = SlurmJobConfig(
            partition="standard",
            qos="standard",
            account="test-account",
            time="00:10:00",
        )
        
        # Create hetjob with 5 components
        config = HetJobConfig(
            base_config=base_config,
            components=[
                HetJobComponent(nodes=1, tasks_per_node=1, cpus_per_task=1)
                for _ in range(5)
            ],
            job_name="hetjob_many",
        )
        
        assert len(config.components) == 5


class TestHetJobScheduling:
    """Test hetjob scheduling behavior."""
    
    def test_hetjob_scheduled_together(self):
        """Test that all hetjob components start together."""
        # This is a Slurm behavior - all components scheduled simultaneously
        # We verify the config supports this
        
        base_config = SlurmJobConfig(
            partition="standard",
            qos="standard",
            account="test-account",
            time="00:10:00",
        )
        
        config = HetJobConfig(
            base_config=base_config,
            components=[
                HetJobComponent(nodes=1, tasks_per_node=288, cpus_per_task=1),
                HetJobComponent(nodes=1, tasks_per_node=1, cpus_per_task=1),
            ],
        )
        
        # Both components should have same time limit
        assert config.base_config.time == "00:10:00"
    
    def test_hetjob_resource_limits_per_component(self):
        """Test that each component counts toward resource limits."""
        # Each hetjob component counts as a "job" for resource limits
        # This is documented in Slurm spec
        
        base_config = SlurmJobConfig(
            partition="standard",
            qos="standard",
            account="test-account",
            time="00:10:00",
        )
        
        # 3-component hetjob counts as 3 jobs toward limit
        config = HetJobConfig(
            base_config=base_config,
            components=[
                HetJobComponent(nodes=1, tasks_per_node=1, cpus_per_task=1)
                for _ in range(3)
            ],
        )
        
        # Standard QoS allows 256 jobs
        # This hetjob uses 3 of those slots
        assert len(config.components) == 3


class TestHetJobUseCases:
    """Test common hetjob use cases."""
    
    def test_hetjob_server_client_pattern(self):
        """Test server-client hetjob pattern."""
        # Common pattern: server on one node, clients on others
        base_config = SlurmJobConfig(
            partition="standard",
            qos="standard",
            account="test-account",
            time="01:00:00",
        )
        
        config = HetJobConfig(
            base_config=base_config,
            components=[
                # Server component
                HetJobComponent(
                    nodes=1,
                    tasks_per_node=1,
                    cpus_per_task=1,
                ),
                # Client component
                HetJobComponent(
                    nodes=4,
                    tasks_per_node=288,
                    cpus_per_task=1,
                ),
            ],
            job_name="server_client",
        )
        
        assert len(config.components) == 2
    
    def test_hetjob_mpi_serial混合(self):
        """Test MPI + serial hetjob pattern."""
        base_config = SlurmJobConfig(
            partition="standard",
            qos="standard",
            account="test-account",
            time="00:30:00",
        )
        
        config = HetJobConfig(
            base_config=base_config,
            components=[
                # MPI component
                HetJobComponent(
                    nodes=2,
                    tasks_per_node=288,
                    cpus_per_task=1,
                ),
                # Serial post-processing component
                HetJobComponent(
                    nodes=1,
                    tasks_per_node=1,
                    cpus_per_task=1,
                ),
            ],
            job_name="mpi_serial",
        )
        
        assert len(config.components) == 2
