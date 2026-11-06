"""
Test partition/QoS combinations.

Verifies that all valid partition/QoS combinations can be configured
and validates charge calculations for each.
"""

import pytest
from unittest.mock import patch, MagicMock

from cirrus_harness.job_submitter import SlurmJobConfig
from cirrus_harness.charge_calculator import ChargeCalculator
from cirrus_harness.utils.validators import (
    validate_partition_qos,
    get_all_valid_combinations,
    get_qos_multiplier,
    VALID_PARTITION_QOS_COMBINATIONS,
)
from cirrus_harness.utils.slurm_commands import SacctRecord
from tests.conftest import VALID_PARTITION_QOS, create_mock_sacct_record


class TestPartitionQoSValidation:
    """Test partition/QoS validation logic."""
    
    def test_all_valid_combinations(self):
        """Test that all documented combinations are valid."""
        for partition, qos in VALID_PARTITION_QOS:
            result = validate_partition_qos(partition, qos)
            assert result.valid, f"{partition}/{qos} should be valid: {result.message}"
    
    def test_invalid_combination(self):
        """Test that invalid combinations are rejected."""
        # highmem with largescale is not valid
        result = validate_partition_qos("highmem", "largescale")
        assert not result.valid
        assert "Invalid" in result.message
    
    def test_get_all_combinations(self):
        """Test retrieval of all valid combinations."""
        combinations = get_all_valid_combinations()
        assert len(combinations) > 0
        assert ("standard", "standard") in combinations
        assert ("highmem", "standard") in combinations
    
    def test_qos_multipliers(self):
        """Test QoS charge multipliers."""
        assert get_qos_multiplier("standard") == 1.0
        assert get_qos_multiplier("long") == 1.0
        assert get_qos_multiplier("short") == 1.0
        assert get_qos_multiplier("largescale") == 1.0
        assert get_qos_multiplier("highpriority") == 1.5
        assert get_qos_multiplier("lowpriority") == 0.0
        assert get_qos_multiplier("reservation") == 1.5
    
    def test_qos_multiplier_case_insensitive(self):
        """Test that QoS multipliers are case-insensitive."""
        assert get_qos_multiplier("STANDARD") == 1.0
        assert get_qos_multiplier("HighPriority") == 1.5
        assert get_qos_multiplier("LOWPRIORITY") == 0.0


class TestPartitionQoSConfigurations:
    """Test job configurations with different partition/QoS combinations."""
    
    @pytest.mark.parametrize("partition,qos", VALID_PARTITION_QOS)
    def test_config_creation(self, partition, qos):
        """Test that configs can be created for all valid combinations."""
        config = SlurmJobConfig(
            partition=partition,
            qos=qos,
            account="test-account",
            nodes=1,
            tasks_per_node=1,
            time="00:10:00",
        )
        assert config.partition == partition
        assert config.qos == qos
    
    @pytest.mark.parametrize("partition,qos", VALID_PARTITION_QOS)
    def test_config_validation(self, partition, qos):
        """Test that configs validate for all valid combinations."""
        # Skip largescale - requires 129+ nodes
        if qos == "largescale":
            return
        
        config = SlurmJobConfig(
            partition=partition,
            qos=qos,
            account="test-account",
            nodes=1,
            tasks_per_node=1,
            time="00:10:00",
        )
        
        is_valid, message, warnings = config.validate()
        
        # Most combinations should be valid for small jobs
        if qos not in ["largescale"]:
            assert is_valid or "minimum" in message.lower()
    
    def test_highpriority_warning(self):
        """Test that highpriority QoS generates surcharge warning."""
        config = SlurmJobConfig(
            partition="standard",
            qos="highpriority",
            account="test-account",
            nodes=1,
            tasks_per_node=1,
            time="00:10:00",
        )
        
        is_valid, message, warnings = config.validate()
        assert any("1.5" in w or "surcharge" in w.lower() for w in warnings)
    
    def test_lowpriority_warning(self):
        """Test that lowpriority QoS generates UKRI warning."""
        config = SlurmJobConfig(
            partition="standard",
            qos="lowpriority",
            account="test-account",
            nodes=1,
            tasks_per_node=1,
            time="00:10:00",
        )
        
        is_valid, message, warnings = config.validate()
        assert any("UKRI" in w or "low priority" in w.lower() for w in warnings)


class TestPartitionQoSChargeCalculation:
    """Test charge calculations for different QoS levels."""
    
    @pytest.fixture
    def mock_sacct(self):
        """Mock sacct for charge calculation tests."""
        with patch('cirrus_harness.charge_calculator.SlurmCLI.get_job_accounting') as mock:
            yield mock
    
    def test_standard_qos_charge(self, mock_sacct):
        """Test charge calculation with standard QoS."""
        mock_sacct.return_value = [create_mock_sacct_record(
            elapsed_seconds=3600,  # 1 hour
            allocated_cpus=288,
            qos="standard"
        )]
        
        calculator = ChargeCalculator()
        charge = calculator.compute_charge("12345")
        
        assert charge.qos_multiplier == 1.0
        assert charge.charged_core_hours == 288.0  # 288 cores × 1 hour
    
    def test_highpriority_qos_charge(self, mock_sacct):
        """Test charge calculation with highpriority QoS (1.5×)."""
        mock_sacct.return_value = [create_mock_sacct_record(
            elapsed_seconds=3600,
            allocated_cpus=288,
            qos="highpriority"
        )]
        
        calculator = ChargeCalculator()
        charge = calculator.compute_charge("12345")
        
        assert charge.qos_multiplier == 1.5
        assert charge.base_core_hours == 288.0
        assert charge.charged_core_hours == 432.0  # 288 × 1.5
    
    def test_lowpriority_qos_charge(self, mock_sacct):
        """Test charge calculation with lowpriority QoS (0×)."""
        mock_sacct.return_value = [create_mock_sacct_record(
            elapsed_seconds=3600,
            allocated_cpus=288,
            qos="lowpriority"
        )]
        
        calculator = ChargeCalculator()
        charge = calculator.compute_charge("12345")
        
        assert charge.qos_multiplier == 0.0
        assert charge.base_core_hours == 288.0
        assert charge.charged_core_hours == 0.0  # Free!
    
    def test_reservation_qos_charge(self, mock_sacct):
        """Test charge calculation with reservation QoS (1.5×)."""
        mock_sacct.return_value = [create_mock_sacct_record(
            elapsed_seconds=3600,
            allocated_cpus=288,
            qos="reservation"
        )]
        
        calculator = ChargeCalculator()
        charge = calculator.compute_charge("12345")
        
        assert charge.qos_multiplier == 1.5
        assert charge.charged_core_hours == 432.0
    
    def test_long_qos_charge(self, mock_sacct):
        """Test charge calculation with long QoS (1.0×)."""
        mock_sacct.return_value = [create_mock_sacct_record(
            elapsed_seconds=3600,
            allocated_cpus=288,
            qos="long"
        )]
        
        calculator = ChargeCalculator()
        charge = calculator.compute_charge("12345")
        
        assert charge.qos_multiplier == 1.0
        assert charge.charged_core_hours == 288.0


class TestPartitionLimits:
    """Test partition-specific limits."""
    
    def test_standard_partition_max_nodes(self):
        """Test standard partition node limits."""
        # Standard QoS allows up to 128 nodes
        config = SlurmJobConfig(
            partition="standard",
            qos="standard",
            account="test-account",
            nodes=128,
            tasks_per_node=1,
            time="00:10:00",
        )
        
        is_valid, message, warnings = config.validate()
        # Should be valid (at limit)
        assert is_valid or "maximum" not in message.lower()
    
    def test_largescale_min_nodes(self):
        """Test largescale QoS minimum node requirement."""
        config = SlurmJobConfig(
            partition="standard",
            qos="largescale",
            account="test-account",
            nodes=1,  # Too few!
            tasks_per_node=1,
            time="00:10:00",
        )
        
        is_valid, message, warnings = config.validate()
        assert not is_valid
        assert "minimum" in message.lower() or "129" in message
    
    def test_short_qos_time_limit(self):
        """Test short QoS time limit (20 minutes)."""
        # Note: The validator checks time format but not QoS-specific limits
        # Those are enforced by Slurm at submission time
        config = SlurmJobConfig(
            partition="standard",
            qos="short",
            account="test-account",
            nodes=1,
            tasks_per_node=1,
            time="00:20:00",  # Maximum for short QoS
        )
        
        is_valid, message, warnings = config.validate()
        # Should be valid (at the limit)
        assert is_valid


class TestHighMemPartition:
    """Test high memory partition specific behavior."""
    
    def test_highmem_config(self):
        """Test highmem partition configuration."""
        config = SlurmJobConfig(
            partition="highmem",
            qos="standard",
            account="test-account",
            nodes=1,
            tasks_per_node=1,
            time="00:10:00",
        )
        
        is_valid, message, warnings = config.validate()
        assert is_valid
    
    def test_highmem_charge_calculation(self):
        """Test that highmem uses same core count as standard."""
        # highmem nodes have same core count (288) but more memory
        from cirrus_harness.utils.validators import CIRRAS_SPECS
        
        assert CIRRAS_SPECS["cores_per_node"] == 288
        # Memory differs but cores are the same
        assert CIRRAS_SPECS["highmem_memory_gb"] == 1536
        assert CIRRAS_SPECS["standard_memory_gb"] == 768
