"""
Configuration Extension for Verification Module
----------------------------------------------

This file extends the base configuration (config.py) with settings
specific to the verification module.

Add these settings to your config.py or import them separately.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Callable, Any


@dataclass
class VerificationConfig:
    """Configuration for the verification module"""
    
    # Tolerance settings
    LOCAL_TOLERANCE: float = 0.1      # Max 10% deviation for numeric fields
    GLOBAL_TOLERANCE: float = 0.05    # Max 5% of records can be modified
    SIGNATURE_THRESHOLD: float = 0.9  # Min 90% watermark bit match
    
    # 【方案B】分级容忍度 - 基于数据集规模动态调整
    ENABLE_TIERED_TOLERANCE: bool = True
    TOLERANCE_SMALL: float = 0.35     # <1000行: 35%
    TOLERANCE_MEDIUM: float = 0.25    # 1000-10000行: 25%
    TOLERANCE_LARGE: float = 0.20     # >10000行: 20%
    
    # Retry settings
    MAX_RETRIES: int = 3              # Maximum reask attempts
    INITIAL_BACKOFF: float = 1.0      # Initial retry delay (seconds)
    BACKOFF_MULTIPLIER: float = 2.0   # Exponential backoff multiplier
    MAX_BACKOFF: float = 10.0         # Maximum retry delay (seconds)
    
    # Feature flags
    ENABLE_REPAIR: bool = True        # Enable automatic repair
    ENABLE_REASK: bool = True         # Enable reask mechanism
    STRICT_VALIDATION: bool = False   # Fail on extra fields
    
    # Schema settings
    AUTO_GENERATE_SCHEMA: bool = True # Auto-generate schema from data
    SCHEMA_PATH: Optional[str] = None # Path to custom schema file
    
    # Repair settings
    REPAIR_STRATEGIES: Optional[Dict[str, Callable]] = None  # Custom repair functions
    
    # Logging settings
    VERIFICATION_LOG_LEVEL: str = "INFO"
    SAVE_VERIFICATION_REPORT: bool = True
    VERIFICATION_REPORT_DIR: str = "parser_data/verification_reports"
    
    # Performance settings
    VALIDATION_BATCH_SIZE: int = 1000  # Batch size for large datasets
    ENABLE_PARALLEL_VALIDATION: bool = False  # Parallel validation (experimental)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            "local_tolerance": self.LOCAL_TOLERANCE,
            "global_tolerance": self.GLOBAL_TOLERANCE,
            "signature_threshold": self.SIGNATURE_THRESHOLD,
            "max_retries": self.MAX_RETRIES,
            "enable_repair": self.ENABLE_REPAIR,
            "enable_reask": self.ENABLE_REASK,
            "strict_validation": self.STRICT_VALIDATION
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'VerificationConfig':
        """Create config from dictionary"""
        return cls(**{k.upper(): v for k, v in config_dict.items() if hasattr(cls, k.upper())})


# Default configuration instance
DEFAULT_VERIFICATION_CONFIG = VerificationConfig()


# Example custom repair strategies
def normalize_email(email: str) -> str:
    """Normalize email addresses"""
    if not isinstance(email, str):
        return email
    return email.lower().strip()


def clip_percentage(value: float) -> float:
    """Clip percentage values to 0-100 range"""
    if not isinstance(value, (int, float)):
        return value
    return max(0.0, min(100.0, float(value)))


def normalize_phone(phone: str) -> str:
    """Normalize phone numbers (remove non-digits)"""
    if not isinstance(phone, str):
        return phone
    return ''.join(filter(str.isdigit, phone))


# Example custom repair strategies dictionary
CUSTOM_REPAIR_STRATEGIES = {
    "email": normalize_email,
    "score": clip_percentage,
    "percentage": clip_percentage,
    "phone": normalize_phone,
    "phone_number": normalize_phone
}


# Schema templates for common data types
SCHEMA_TEMPLATES = {
    "user_profile": {
        "id": {"type": int, "min": 0},
        "name": {"type": str, "default": "Unknown"},
        "email": {"type": str, "default": "unknown@example.com"},
        "age": {"type": int, "min": 0, "max": 150},
        "active": {"type": bool, "default": True}
    },
    "transaction": {
        "transaction_id": {"type": str},
        "amount": {"type": float, "min": 0.0},
        "timestamp": {"type": str},
        "status": {"type": str, "default": "pending"}
    },
    "log_entry": {
        "timestamp": {"type": str},
        "level": {"type": str, "default": "INFO"},
        "message": {"type": str},
        "source": {"type": str, "default": "unknown"}
    }
}


def get_verification_config_for_dataset(dataset_name: str) -> VerificationConfig:
    """
    Get recommended verification config for specific dataset types
    
    Parameters
    ----------
    dataset_name : str
        Name of the dataset type (e.g., 'user_data', 'financial', 'logs')
        
    Returns
    -------
    VerificationConfig
        Recommended configuration
    """
    if dataset_name.lower() in ['financial', 'transaction', 'banking']:
        # Strict settings for financial data
        return VerificationConfig(
            LOCAL_TOLERANCE=0.01,      # 1% max deviation
            GLOBAL_TOLERANCE=0.02,     # 2% max modifications
            SIGNATURE_THRESHOLD=0.95,  # 95% signature match
            MAX_RETRIES=5,
            STRICT_VALIDATION=True
        )
    
    elif dataset_name.lower() in ['logs', 'system_logs', 'event_logs']:
        # More lenient for log data
        return VerificationConfig(
            LOCAL_TOLERANCE=0.15,      # 15% max deviation
            GLOBAL_TOLERANCE=0.1,      # 10% max modifications
            SIGNATURE_THRESHOLD=0.85,  # 85% signature match
            MAX_RETRIES=2,
            STRICT_VALIDATION=False
        )
    
    elif dataset_name.lower() in ['user', 'customer', 'profile']:
        # Balanced settings for user data
        return VerificationConfig(
            LOCAL_TOLERANCE=0.1,       # 10% max deviation
            GLOBAL_TOLERANCE=0.05,     # 5% max modifications
            SIGNATURE_THRESHOLD=0.9,   # 90% signature match
            MAX_RETRIES=3,
            STRICT_VALIDATION=True
        )
    
    else:
        # Default settings
        return DEFAULT_VERIFICATION_CONFIG


# Integration example for config.py
INTEGRATION_EXAMPLE = """
# Add to config.py:

from verification_config import VerificationConfig, DEFAULT_VERIFICATION_CONFIG

class CFG:
    # ... existing config ...
    
    # Verification settings
    VERIFICATION = DEFAULT_VERIFICATION_CONFIG
    
    # Or customize:
    # VERIFICATION = VerificationConfig(
    #     LOCAL_TOLERANCE=0.1,
    #     GLOBAL_TOLERANCE=0.05,
    #     SIGNATURE_THRESHOLD=0.9,
    #     MAX_RETRIES=3,
    #     ENABLE_REPAIR=True,
    #     ENABLE_REASK=True
    # )
"""

if __name__ == "__main__":
    # Print configuration examples
    print("Default Verification Configuration:")
    print("=" * 60)
    for key, value in DEFAULT_VERIFICATION_CONFIG.to_dict().items():
        print(f"  {key}: {value}")
    
    print("\n\nFinancial Data Configuration:")
    print("=" * 60)
    financial_config = get_verification_config_for_dataset('financial')
    for key, value in financial_config.to_dict().items():
        print(f"  {key}: {value}")
    
    print("\n\nLog Data Configuration:")
    print("=" * 60)
    log_config = get_verification_config_for_dataset('logs')
    for key, value in log_config.to_dict().items():
        print(f"  {key}: {value}")
    
    print("\n\nIntegration Example:")
    print("=" * 60)
    print(INTEGRATION_EXAMPLE)
