from typing import Dict, Optional

class FeeCalculator:
    # Fee tiers based on common exchange fee structures
    FEE_TIER_MAP: Dict[str, float] = {
        "VIP0": 0.0010,    # 0.10% - Base tier
        "VIP1": 0.0008,    # 0.08% 
        "VIP2": 0.0006,    # 0.06%
        "VIP3": 0.00045,   # 0.045%
        "VIP4": 0.00035,   # 0.035%
        "VIP5": 0.00025,   # 0.025%
        # Special tiers
        "Market Maker": 0.00020,  # 0.02% - Market maker tier
        "Institutional": 0.00015  # 0.015% - Institutional tier
    }
    
    DEFAULT_FEE_RATE = 0.0010  # Default to base tier (0.10%)
    
    @classmethod
    def get_fee_rate(cls, fee_tier: Optional[str] = None) -> float:
        """
        Get the fee rate for a given tier.
        
        Args:
            fee_tier: The fee tier string identifier
            
        Returns:
            float: The fee rate for the tier (as a decimal, e.g., 0.001 for 0.1%)
        """
        if not fee_tier:
            return cls.DEFAULT_FEE_RATE
            
        # Normalize the tier string to handle case variations
        normalized_tier = fee_tier.strip().upper()
        
        # Try to find the exact tier
        if normalized_tier in cls.FEE_TIER_MAP:
            return cls.FEE_TIER_MAP[normalized_tier]
            
        # Try to find by partial match (e.g., "VIP 1" matches "VIP1")
        normalized_tier = normalized_tier.replace(" ", "")
        for tier, rate in cls.FEE_TIER_MAP.items():
            if tier.replace(" ", "") == normalized_tier:
                return rate
        
        # Return default if no match found
        return cls.DEFAULT_FEE_RATE
        
    @classmethod
    def calculate_fee(cls, quantity: float, fee_tier: Optional[str] = None) -> float:
        """
        Calculate the fee for a given quantity and tier.
        
        Args:
            quantity: The order quantity
            fee_tier: The fee tier string identifier
            
        Returns:
            float: The calculated fee amount
        """
        fee_rate = cls.get_fee_rate(fee_tier)
        return quantity * fee_rate 