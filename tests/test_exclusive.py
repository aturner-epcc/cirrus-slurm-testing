"""
Test exclusive vs non-exclusive job behavior.

Verifies that --exclusive correctly affects charge calculations
by charging for entire nodes regardless of actual core usage.
"""

import pytest
from unittest.mock import patch

from cirrus_harness.job_submitter import SlurmJobConfig
from cirrus_harness.charge_calculator import ChargeCalculator
from cirrus_harness.utils.validators import CIRRAS_SPECS
from tests.conftest import create_mock_sacct_record


class TestExclusiveConfiguration:
    """Test exclusive access configuration."""
    
    def test_exclusive_flag_set(self):
        """Test that exclusive flag is properly set."""
        config = SlurmJobConfig(
            partition="standard",
            qos="standard",
            account="test-account",
            nodes=1,
            tasks_per_node=1,
            cpus_per_task=1,
            time="00:10:00",
            exclusive=True,
        )
        
        assert config.exclusive is True
    
    def test_non_exclusive_flag_set(self):
        """Test non-exclusive configuration."""
        config = SlurmJobConfig(
            partition="standard",
            qos="standard",
            account="test-account",
            nodes=1,
            tasks_per_node=1,
            cpus_per_task=1,
            time="00:10:00",
            exclusive=False,
        )
        
        assert config.exclusive is False
    
    def test_exclusive_cores_calculation(self):
        """Test that exclusive jobs charge for full node."""
        config = SlurmJobConfig(
            partition="standard",
            qos="standard",
            account="test-account",
            nodes=1,
            tasks_per_node=1,  # Only 1 core requested
            cpus_per_task=1,
            time="00:10:00",
            exclusive=True,
        )
        
        assert config.total_cores == 1  # Actually using 1 core
        assert config.effective_cores == 288  # Charged for 288
    
    def test_exclusive_multinode_calculation(self):
        """Test exclusive with multiple nodes."""
        config = SlurmJobConfig(
            partition="standard",
            qos="standard",
            account="test-account",
            nodes=4,
            tasks_per_node=64,  # Partial node usage
            cpus_per_task=1,
            time="00:20:00",
            exclusive=True,
        )
        
        assert config.total_cores == 256  # 4 × 64
        assert config.effective_cores == 1152  # 4 × 288 (full nodes)


class TestExclusiveChargeCalculation:
    """Test charge calculations with exclusive access."""
    
    def test_exclusive_single_core_full_node_charge(self):
        """Test that 1 core with --exclusive charges for full node."""
        with patch('cirrus_harness.charge_calculator.SlurmCLI.get_job_accounting') as mock_sacct:
            # Job requested 1 core but with exclusive
            mock_sacct.return_value = [create_mock_sacct_record(
                elapsed_seconds=3600,  # 1 hour
                allocated_cpus=1,
                allocated_nodes=1,
                qos="standard"
            )]
            
            config = SlurmJobConfig(
                partition="standard",
                qos="standard",
                account="test-account",
                nodes=1,
                tasks_per_node=1,
                cpus_per_task=1,
                time="01:00:00",
                exclusive=True,
            )
            
            calculator = ChargeCalculator(default_exclusive=True)
            charge = calculator.compute_charge("12345", config)
            
            # Should charge for 288 cores, not 1
            assert charge.allocated_cores == 1  # What was requested
            assert charge.effective_cores == 288  # What we're charged for
            assert charge.base_core_hours == 288.0  # 288 cores × 1 hour
    
    def test_non_exclusive_single_core_charge(self):
        """Test that 1 core without --exclusive charges for 1 core."""
        with patch('cirrus_harness.charge_calculator.SlurmCLI.get_job_accounting') as mock_sacct:
            mock_sacct.return_value = [create_mock_sacct_record(
                elapsed_seconds=3600,
                allocated_cpus=1,
                allocated_nodes=1,
                qos="standard"
            )]
            
            config = SlurmJobConfig(
                partition="standard",
                qos="standard",
                account="test-account",
                nodes=1,
                tasks_per_node=1,
                cpus_per_task=1,
                time="01:00:00",
                exclusive=False,
            )
            
            calculator = ChargeCalculator(default_exclusive=False)
            charge = calculator.compute_charge("12345", config)
            
            assert charge.allocated_cores == 1
            assert charge.effective_cores == 1
            assert charge.base_core_hours == 1.0
    
    def test_exclusive_partial_node_charge(self):
        """Test exclusive with partial node usage."""
        with patch('cirrus_harness.charge_calculator.SlurmCLI.get_job_accounting') as mock_sacct:
            mock_sacct.return_value = [create_mock_sacct_record(
                elapsed_seconds=3600,
                allocated_cpus=64,  # Using 64 cores
                allocated_nodes=1,
                qos="standard"
            )]
            
            config = SlurmJobConfig(
                partition="standard",
                qos="standard",
                account="test-account",
                nodes=1,
                tasks_per_node=64,
                cpus_per_task=1,
                time="01:00:00",
                exclusive=True,
            )
            
            calculator = ChargeCalculator(default_exclusive=True)
            charge = calculator.compute_charge("12345", config)
            
            # Charged for full node despite using only 64 cores
            assert charge.effective_cores == 288
            assert charge.base_core_hours == 288.0
    
    def test_exclusive_multinode_partial_usage(self):
        """Test exclusive with multiple nodes, partial usage per node."""
        with patch('cirrus_harness.charge_calculator.SlurmCLI.get_job_accounting') as mock_sacct:
            mock_sacct.return_value = [create_mock_sacct_record(
                elapsed_seconds=3600,
                allocated_cpus=256,  # 4 nodes × 64 cores
                allocated_nodes=4,
                qos="standard"
            )]
            
            config = SlurmJobConfig(
                partition="standard",
                qos="standard",
                account="test-account",
                nodes=4,
                tasks_per_node=64,
                cpus_per_task=1,
                time="01:00:00",
                exclusive=True,
            )
            
            calculator = ChargeCalculator(default_exclusive=True)
            charge = calculator.compute_charge("12345", config)
            
            # Charged for 4 full nodes
            assert charge.effective_cores == 1152  # 4 × 288
            assert charge.base_core_hours == 1152.0
    
    def test_exclusive_charge_ratio(self):
        """Test the charge ratio between exclusive and non-exclusive."""
        with patch('cirrus_harness.charge_calculator.SlurmCLI.get_job_accounting') as mock_sacct:
            # Same job, different exclusive settings
            mock_sacct.return_value = [create_mock_sacct_record(
                elapsed_seconds=3600,
                allocated_cpus=1,
                allocated_nodes=1,
                qos="standard"
            )]
            
            calculator_exclusive = ChargeCalculator(default_exclusive=True)
            calculator_non_exclusive = ChargeCalculator(default_exclusive=False)
            
            charge_exclusive = calculator_exclusive.compute_charge("12345")
            charge_non_exclusive = calculator_non_exclusive.compute_charge("12346")
            
            # Exclusive should charge 288× more for single core job
            ratio = charge_exclusive.base_core_hours / charge_non_exclusive.base_core_hours
            assert ratio == 288.0


class TestExclusiveScriptGeneration:
    """Test that exclusive flag appears in scripts."""
    
    def test_exclusive_in_script(self, temp_output_dir):
        """Test that --exclusive appears in generated script."""
        from cirrus_harness.job_submitter import JobSubmitter
        
        config = SlurmJobConfig(
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
        
        submitter = JobSubmitter(output_dir=temp_output_dir)
        script_path = submitter.generate_script(config, ["hostname"], job_type="serial")
        
        content = script_path.read_text()
        assert "#SBATCH --exclusive" in content
    
    def test_non_exclusive_not_in_script(self, temp_output_dir):
        """Test that --exclusive doesn't appear when not set."""
        from cirrus_harness.job_submitter import JobSubmitter
        
        config = SlurmJobConfig(
            partition="standard",
            qos="standard",
            account="test-account",
            nodes=1,
            tasks_per_node=1,
            cpus_per_task=1,
            time="00:10:00",
            exclusive=False,
            job_name="non_exclusive_test",
        )
        
        submitter = JobSubmitter(output_dir=temp_output_dir)
        script_path = submitter.generate_script(config, ["hostname"], job_type="serial")
        
        content = script_path.read_text()
        assert "#SBATCH --exclusive" not in content
    
    def test_multinode_auto_exclusive_warning(self):
        """Test that multi-node jobs warn about exclusive."""
        config = SlurmJobConfig(
            partition="standard",
            qos="standard",
            account="test-account",
            nodes=2,
            tasks_per_node=1,
            cpus_per_task=1,
            time="00:10:00",
            exclusive=False,
        )
        
        is_valid, message, warnings = config.validate()
        
        # Should warn that multi-node jobs should use exclusive
        assert any("exclusive" in w.lower() for w in warnings)


class TestExclusiveWarningMessages:
    """Test that appropriate warnings are generated for exclusive usage."""
    
    def test_exclusive_partial_node_warning(self):
        """Test warning when using exclusive with partial node."""
        config = SlurmJobConfig(
            partition="standard",
            qos="standard",
            account="test-account",
            nodes=1,
            tasks_per_node=1,
            cpus_per_task=1,
            time="00:10:00",
            exclusive=True,
        )
        
        is_valid, message, warnings = config.validate()
        
        # Should warn about charging for full node
        assert any("288" in w for w in warnings)
    
    def test_exclusive_warning_includes_charge_info(self):
        """Test that warning mentions charging implications."""
        config = SlurmJobConfig(
            partition="standard",
            qos="standard",
            account="test-account",
            nodes=1,
            tasks_per_node=10,
            cpus_per_task=1,
            time="00:10:00",
            exclusive=True,
        )
        
        is_valid, message, warnings = config.validate()
        
        # Warning should mention charging for all cores
        assert any("charged" in w.lower() for w in warnings)


class TestExclusiveBestPractices:
    """Test exclusive access best practices."""
    
    def test_full_node_no_warning(self):
        """Test that full node usage with exclusive doesn't warn."""
        config = SlurmJobConfig(
            partition="standard",
            qos="standard",
            account="test-account",
            nodes=1,
            tasks_per_node=288,
            cpus_per_task=1,
            time="00:10:00",
            exclusive=True,
        )
        
        is_valid, message, warnings = config.validate()
        
        # Should not warn about exclusive when using full node
        assert not any("288" in w and "charged" in w.lower() for w in warnings)
    
    def test_mpi_multinode_exclusive_required(self):
        """Test that multi-node MPI jobs should use exclusive."""
        config = SlurmJobConfig(
            partition="standard",
            qos="standard",
            account="test-account",
            nodes=4,
            tasks_per_node=288,
            cpus_per_task=1,
            time="01:00:00",
            exclusive=False,
        )
        
        is_valid, message, warnings = config.validate()
        
        # Should warn about needing exclusive for multi-node
        assert any("exclusive" in w.lower() for w in warnings)
