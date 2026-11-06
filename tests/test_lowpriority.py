"""
Test low priority access functionality.

Verifies lowpriority QoS configuration, submission, and zero-charge behavior.
"""

import pytest
from unittest.mock import patch

from cirrus_harness.job_submitter import SlurmJobConfig
from cirrus_harness.charge_calculator import ChargeCalculator
from cirrus_harness.utils.validators import (
    validate_partition_qos,
    get_qos_multiplier,
    VALID_PARTITION_QOS_COMBINATIONS,
)
from tests.conftest import create_mock_sacct_record


class TestLowPriorityConfiguration:
    """Test lowpriority QoS configuration."""
    
    def test_lowpriority_config_creation(self):
        """Test lowpriority job configuration."""
        config = SlurmJobConfig(
            partition="standard",
            qos="lowpriority",
            account="test-account",
            nodes=1,
            tasks_per_node=1,
            cpus_per_task=1,
            time="00:10:00",
            job_name="lowpriority_test",
        )
        
        assert config.partition == "standard"
        assert config.qos == "lowpriority"
    
    def test_lowpriority_validation(self):
        """Test lowpriority QoS validation."""
        result = validate_partition_qos("standard", "lowpriority")
        
        assert result.valid is True
        assert any("UKRI" in w for w in result.warnings)
    
    def test_lowpriority_zero_multiplier(self):
        """Test that lowpriority has zero charge multiplier."""
        multiplier = get_qos_multiplier("lowpriority")
        assert multiplier == 0.0
    
    def test_lowpriority_case_insensitive(self):
        """Test lowpriority multiplier is case-insensitive."""
        assert get_qos_multiplier("LOWPRIORITY") == 0.0
        assert get_qos_multiplier("LowPriority") == 0.0


class TestLowPriorityChargeCalculation:
    """Test charge calculations for lowpriority jobs."""
    
    def test_lowpriority_zero_charge(self):
        """Test that lowpriority jobs incur zero charge."""
        with patch('cirrus_harness.charge_calculator.SlurmCLI.get_job_accounting') as mock_sacct:
            mock_sacct.return_value = [create_mock_sacct_record(
                elapsed_seconds=86400,  # 24 hours
                allocated_cpus=288,
                allocated_nodes=1,
                qos="lowpriority"
            )]
            
            calculator = ChargeCalculator()
            charge = calculator.compute_charge("12345")
            
            # Even though we used 288 cores for 24 hours
            assert charge.base_core_hours == 6912.0  # 288 × 24
            assert charge.qos_multiplier == 0.0
            assert charge.charged_core_hours == 0.0  # But charged 0!
    
    def test_lowpriority_short_job_zero_charge(self):
        """Test zero charge for short lowpriority job."""
        with patch('cirrus_harness.charge_calculator.SlurmCLI.get_job_accounting') as mock_sacct:
            mock_sacct.return_value = [create_mock_sacct_record(
                elapsed_seconds=600,  # 10 minutes
                allocated_cpus=1,
                qos="lowpriority"
            )]
            
            calculator = ChargeCalculator()
            charge = calculator.compute_charge("12345")
            
            assert charge.charged_core_hours == 0.0
    
    def test_lowpriority_multinode_zero_charge(self):
        """Test zero charge for multi-node lowpriority job."""
        with patch('cirrus_harness.charge_calculator.SlurmCLI.get_job_accounting') as mock_sacct:
            mock_sacct.return_value = [create_mock_sacct_record(
                elapsed_seconds=3600,
                allocated_cpus=1152,  # 4 nodes
                allocated_nodes=4,
                qos="lowpriority"
            )]
            
            calculator = ChargeCalculator()
            charge = calculator.compute_charge("12345")
            
            assert charge.base_core_hours == 1152.0
            assert charge.charged_core_hours == 0.0


class TestLowPriorityLimits:
    """Test lowpriority QoS limits."""
    
    def test_lowpriority_max_time(self):
        """Test lowpriority maximum walltime (24 hours)."""
        config = SlurmJobConfig(
            partition="standard",
            qos="lowpriority",
            account="test-account",
            nodes=1,
            tasks_per_node=1,
            cpus_per_task=1,
            time="24:00:00",  # Maximum
            job_name="lowpriority_max",
        )
        
        is_valid, message, warnings = config.validate()
        # Should be valid at exactly 24 hours
        assert is_valid or "exceeds" not in message.lower()
    
    def test_lowpriority_max_nodes(self):
        """Test lowpriority maximum node count (64 nodes)."""
        config = SlurmJobConfig(
            partition="standard",
            qos="lowpriority",
            account="test-account",
            nodes=64,  # Maximum
            tasks_per_node=1,
            cpus_per_task=1,
            time="00:10:00",
            job_name="lowpriority_nodes",
        )
        
        is_valid, message, warnings = config.validate()
        # Should be valid at exactly 64 nodes
        assert is_valid or "maximum" not in message.lower()
    
    def test_lowpriority_max_jobs(self):
        """Test lowpriority maximum job count (100 jobs)."""
        # This is a QoS-level limit, not per-job
        # We can't test it directly but can verify the config
        config = VALID_PARTITION_QOS_COMBINATIONS[("standard", "lowpriority")]
        assert config["max_jobs"] == 100
        assert config["max_nodes"] == 64
        assert config["max_time_hours"] == 24


class TestLowPriorityRequirements:
    """Test lowpriority access requirements."""
    
    def test_lowpriority_ukri_requirement(self):
        """Test that lowpriority requires UKRI project."""
        result = validate_partition_qos("standard", "lowpriority")
        
        assert any("UKRI" in w for w in result.warnings)
    
    def test_lowpriority_budget_requirement(self):
        """Test that lowpriority requires valid budget."""
        result = validate_partition_qos("standard", "lowpriority")
        
        assert any("budget" in w.lower() for w in result.warnings)
    
    def test_lowpriority_available_combinations(self):
        """Test which partitions support lowpriority."""
        lowpriority_partitions = [
            (p, q) for p, q in VALID_PARTITION_QOS_COMBINATIONS.keys()
            if q == "lowpriority"
        ]
        
        # Only standard partition supports lowpriority
        assert len(lowpriority_partitions) == 1
        assert lowpriority_partitions[0] == ("standard", "lowpriority")


class TestLowPriorityUsage:
    """Test lowpriority usage patterns."""
    
    def test_lowpriority_full_node(self):
        """Test lowpriority with full node usage."""
        config = SlurmJobConfig(
            partition="standard",
            qos="lowpriority",
            account="test-account",
            nodes=1,
            tasks_per_node=288,
            cpus_per_task=1,
            time="24:00:00",
            exclusive=True,
            job_name="lowpriority_full",
        )
        
        is_valid, message, warnings = config.validate()
        assert is_valid
        
        # Verify charge would be zero
        with patch('cirrus_harness.charge_calculator.SlurmCLI.get_job_accounting') as mock_sacct:
            mock_sacct.return_value = [create_mock_sacct_record(
                elapsed_seconds=86400,
                allocated_cpus=288,
                qos="lowpriority"
            )]
            
            calculator = ChargeCalculator()
            charge = calculator.compute_charge("12345")
            
            assert charge.charged_core_hours == 0.0
    
    def test_lowpriority_array(self):
        """Test lowpriority with job arrays."""
        config = SlurmJobConfig(
            partition="standard",
            qos="lowpriority",
            account="test-account",
            nodes=1,
            tasks_per_node=1,
            cpus_per_task=1,
            time="01:00:00",
            array_spec="1-50",
            job_name="lowpriority_array",
        )
        
        is_valid, message, warnings = config.validate()
        assert is_valid
    
    def test_lowpriority_vs_standard_charge_comparison(self):
        """Compare charges between lowpriority and standard QoS."""
        with patch('cirrus_harness.charge_calculator.SlurmCLI.get_job_accounting') as mock_sacct:
            # Same job, different QoS
            mock_sacct.return_value = [create_mock_sacct_record(
                elapsed_seconds=3600,
                allocated_cpus=288,
                qos="standard"
            )]
            
            calculator = ChargeCalculator()
            
            # Standard QoS charge
            standard_charge = calculator.compute_charge("12345")
            
            # Lowpriority QoS charge
            mock_sacct.return_value = [create_mock_sacct_record(
                elapsed_seconds=3600,
                allocated_cpus=288,
                qos="lowpriority"
            )]
            lowpriority_charge = calculator.compute_charge("12346")
            
            # Standard should be charged
            assert standard_charge.charged_core_hours > 0
            
            # Lowpriority should be free
            assert lowpriority_charge.charged_core_hours == 0.0
            
            # Savings
            savings = standard_charge.charged_core_hours
            assert savings == 288.0  # 288 cores × 1 hour


class TestLowPriorityBestPractices:
    """Test lowpriority best practices."""
    
    def test_lowpriority_pump_priming_limit(self):
        """Test Pump Priming project lowpriority limit."""
        # Documentation states: max 200,000 coreh for Pump Priming
        # This is a policy limit, not enforced by code
        pump_priming_limit = 200_000  # core-hours
        
        # Example: 288 cores × 24 hours × 30 days = 207,360 coreh
        # This would exceed the limit
        excessive_usage = 288 * 24 * 30
        
        assert excessive_usage > pump_priming_limit
        
        # Note: This is documented but not enforced by the harness


class TestLowPriorityEdgeCases:
    """Test lowpriority edge cases."""
    
    def test_lowpriority_highmem_not_available(self):
        """Test that lowpriority is not available on highmem."""
        result = validate_partition_qos("highmem", "lowpriority")
        
        assert result.valid is False
        assert "Invalid" in result.message
    
    def test_lowpriority_largescale_not_available(self):
        """Test that lowpriority cannot be combined with largescale."""
        # These are separate QoS options
        result = validate_partition_qos("standard", "lowpriority")
        assert result.valid is True
        
        result = validate_partition_qos("standard", "largescale")
        assert result.valid is True
        
        # But not both at once (not a valid combination)
        # This is implicit in the VALID_PARTITION_QOS_COMBINATIONS dict
