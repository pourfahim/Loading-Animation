import asyncio
import logging
from logging.handlers import RotatingFileHandler
import subprocess
import hashlib
import ed25519
import requests
import random
import time
import re
import yaml
import os
import sys
import signal
from decimal import Decimal
from typing import Dict, List, Optional, Tuple, Union, Any  # Added Tuple here
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
import base64
import warnings
import hmac
import aiohttp
import json

FLOWX_CONSTANTS = {
    'TESTNET': {
        'PACKAGE_ID': '0x1812951b018fcd9f2262d160acdaff5b432011c58fae80eb981c7be3370167da',
        'POOL_REGISTRY': '0x71a69d9d7319c0c86e0a5266746f85481840064e19fdb491ce83843851f5fe9d'
    },
    'DEFAULT_SLIPPAGE': 0.5,    # 0.5% slippage
    'DEFAULT_DEADLINE': 300,     # 5 minutes
    'DEBUG_MODE': True          # Enable detailed debugging
}

@dataclass
class FlowXSwapParams:
    """Parameters required for FlowX swap transaction"""
    sender: str
    coin_in: str
    coin_out: str
    amount_in: int
    slippage: Decimal
    deadline: int
    pool_id: str

# Ignore urllib3 warnings
warnings.filterwarnings("ignore", category=Warning)

# Create logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create rotating file handler
file_handler = RotatingFileHandler(
    'sui_bot.log',
    maxBytes=10*1024*1024,  # 10MB
    backupCount=5
)

# Create formatter and add it to the handler
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# Add handler to logger
logger.addHandler(file_handler)

# Remove existing handlers to avoid duplicate logs
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# Add this after your imports but before NETWORKS configuration
DEADLINE_DURATION = 30  # 30 seconds for transaction deadline

# Slippage Configurations
SLIPPAGE_PRESETS = {
    "1": {  # Conservative
        "name": "Conservative (Safer, for stable pairs/high liquidity)",
        "min": 0.5,  # 0.5%
        "max": 1.0   # 1.0%
    },
    "2": {  # Moderate
        "name": "Moderate (Balanced, most common)",
        "min": 1.0,  # 1.0%
        "max": 2.0   # 2.0%
    },
    "3": {  # Aggressive
        "name": "Aggressive (For low liquidity/volatile pairs)",
        "min": 2.0,  # 2.0%
        "max": 3.0   # 3.0%
    },
    "4": {  # Custom
        "name": "Custom (Set your own slippage)",
        "min": None,
        "max": None
    }
}

# Slippage Safety Limits
MIN_ALLOWED_SLIPPAGE = 0.1    # 0.1% minimum allowed
MAX_ALLOWED_SLIPPAGE = 49.9   # 49.9% maximum allowed
HIGH_SLIPPAGE_WARNING = 5.0   # Warning threshold
EXTREME_SLIPPAGE_WARNING = 10.0  # Extreme warning threshold

# Display Configuration for Slippage Options
SLIPPAGE_DISPLAY = """
=== Slippage Configuration Options ===

1. Conservative (0.5% - 1.0%)
   • Safest option for trading
   • Recommended for:
     - High liquidity token pairs
     - Stable token pairs
     - When minimizing price impact is priority
   • Min Slippage: 0.5%
   • Max Slippage: 1.0%

2. Moderate (1.0% - 2.0%)
   • Balanced option for most trades
   • Recommended for:
     - Medium liquidity pairs
     - Standard trading pairs
     - Regular trading conditions
   • Min Slippage: 1.0%
   • Max Slippage: 2.0%

3. Aggressive (2.0% - 3.0%)
   • Higher risk option
   • Recommended for:
     - Low liquidity pairs
     - Volatile trading pairs
     - When trade execution speed is priority
   • Min Slippage: 2.0%
   • Max Slippage: 3.0%

4. Custom Slippage
   • Set your own slippage range
   • You will need to input:
     - Minimum slippage percentage
     - Maximum slippage percentage
   • Note: Higher slippage increases execution chance but also price impact risk
"""

# Network Configuration with Multiple Endpoints
NETWORKS = {
    "testnet": {
        "sui_official": {
            "url": "https://sui-testnet.publicnode.com",
            "headers": {
                "Content-Type": "application/json",
                "Accept": "application/json"
            },
            "priority": 1
        },
        "blockvision_1": {
            "url": "https://sui-testnet.blockvision.org/v1/2o9ltACTt2ZpHLoVJDF0lCezeRH",
            "api_key": "2o9ltACTt2ZpHLoVJDF0lCezeRH",
            "headers": {
                "X-API-Key": "2o9ltACTt2ZpHLoVJDF0lCezeRH",
                "Content-Type": "application/json"
            },
            "priority": 2
        },
        "blockvision_2": {
            "url": "https://sui-testnet.blockvision.org/v1/2oA1qAQx4vzKvDgnqR1ca2e6sk3",
            "api_key": "2oA1qAQx4vzKvDgnqR1ca2e6sk3",
            "headers": {
                "X-API-Key": "2oA1qAQx4vzKvDgnqR1ca2e6sk3",
                "Content-Type": "application/json"
            },
            "priority": 2
        },
        "ankr": {
            "url": "https://rpc.ankr.com/sui_testnet/6825e4e1cde85e00de078ffe5a4f02caebf4858cb02ced2363212e0256b3c4f5",
            "headers": {
                "Content-Type": "application/json"
            },
            "priority": 3
        }
    },
    "mainnet": {
        "sui_official": {
            "url": "https://sui-mainnet.publicnode.com",
            "headers": {
                "Content-Type": "application/json",
                "Accept": "application/json"
            },
            "priority": 1
        },
        "blockvision_1": {
            "url": "https://sui-mainnet.blockvision.org/v1/2o9yNjIQF0oWAtEktYVYfvl9cAU",
            "api_key": "2o9yNjIQF0oWAtEktYVYfvl9cAU",
            "headers": {
                "X-API-Key": "2o9yNjIQF0oWAtEktYVYfvl9cAU",
                "Content-Type": "application/json"
            },
            "priority": 2
        },
        "blockvision_2": {
            "url": "https://sui-mainnet.blockvision.org/v1/2oA1oqG4c6KfUTGVmbvmjryQbHf",
            "api_key": "2oA1oqG4c6KfUTGVmbvmjryQbHf",
            "headers": {
                "X-API-Key": "2oA1oqG4c6KfUTGVmbvmjryQbHf",
                "Content-Type": "application/json"
            },
            "priority": 2
        },
        "blockvision_3": {
            "url": "https://sui-mainnet.blockvision.org/v1/2oA3IjAbROXejSsca3v1nIN0AIn",
            "api_key": "2oA3IjAbROXejSsca3v1nIN0AIn",
            "headers": {
                "X-API-Key": "2oA3IjAbROXejSsca3v1nIN0AIn",
                "Content-Type": "application/json"
            },
            "priority": 2
        },
        "allthatnode_1": {
            "url": "https://sui-mainnet.g.allthatnode.com/full/json_rpc/3614cbbb9b784f168d8f3ad3960e9941",
            "headers": {
                "Content-Type": "application/json"
            },
            "priority": 3
        },
        "allthatnode_2": {
            "url": "https://sui-mainnet.g.allthatnode.com/full/json_rpc/12465ca193604d68b806bd7a2236732d",
            "headers": {
                "Content-Type": "application/json"
            },
            "priority": 3
        }
    }
}

# Usage tracking with updated limits
USAGE_LIMITS = {
    "sui_official": {
        "daily_limit": 1000000,  # Higher limit for official endpoints
        "request_cost": 1
    },
    "blockvision": {
        "daily_limit": 200000,
        "request_cost": 50
    },
    "allthatnode": {
        "daily_limit": 10000,
        "request_cost": 50
    },
    "ankr": {
        "daily_limit": 100000,
        "request_cost": 50
    }
}

class SuiConstants:
    """Constants for Sui blockchain interaction"""
    # Gas constants
    GAS_BUFFER_FACTOR = 1.1
    MIN_GAS_BUDGET = 2_000_000  # 2 SUI in MIST
    MAX_GAS_BUDGET = 50_000_000_000
    DEFAULT_GAS_BUDGET = 20_000_000
    MIST_PER_SUI = 1_000_000_000
    
    # Token constants
    COIN_TYPE_SUI = "0x2::sui::SUI"
    DEFAULT_GAS_PRICE = 1000
    
    # Transaction constants
    SIGNATURE_SCHEME = "ED25519"
    INTENT_SCOPE = bytes([0])
    INTENT_VERSION = bytes([0])
    INTENT_APP_ID = bytes([0])

@dataclass
class BotConfig:
    """Bot configuration parameters"""
    network_type: str
    token_contract: str
    package_id: str
    base_amount: Decimal
    fluctuation_range: Decimal
    min_interval: int
    max_interval: int
    run_duration: int
    fee_percentage: Decimal
    num_wallets: int
    gas_budget: int = 2_000_000  # 2 SUI in MIST as default
    slippage_tolerance: Decimal = Decimal('0.01')

@dataclass
class NetworkConfig:
    """Network configuration parameters"""
    network_type: str
    rpc_url: str
    api_key: Optional[str] = None
    provider_type: Optional[str] = None

@dataclass
class SuiTransaction:
    """Represents a Sui transaction"""
    tx_bytes: str
    signature: Optional[str] = None
    public_key: Optional[str] = None
    tx_digest: Optional[str] = None
    status: str = "pending"
    gas_used: int = 0
    timestamp: float = field(default_factory=time.time)

class SuiTransactionBuilder:
    """Handles building and signing SUI transactions"""
    
    @staticmethod
    def get_intent_bytes() -> bytes:
        """Get correctly formatted intent bytes"""
        return bytes([0]) + bytes([0]) + bytes([0])  # scope + version + app_id

    def __init__(self, network_manager: "NetworkManager"):
        self.network_manager = network_manager
        self.MIST_PER_SUI = 1000000000
        
    async def build_transfer_tx(self, sender_address: str, coin_object_id: str, 
                              recipient: str, amount_mist: int, gas_budget: int) -> str:
        """Build a SUI transfer transaction"""
        try:
            tx_params = [
                sender_address,
                [coin_object_id],
                [recipient],
                [str(amount_mist)],
                str(gas_budget)
            ]
            
            response = await self.network_manager.make_request(
                "unsafe_paySui",
                tx_params
            )
            
            return response['txBytes']
            
        except Exception as e:
            logger.error(f"Failed to build transfer transaction: {str(e)}")
            raise

    async def build_trade_tx(self, sender_address: str, contract: str, 
                           function: str, amount_mist: int, gas_budget: int) -> str:
        """Build a trade transaction"""
        try:
            # Get gas price
            gas_price = await self._get_reference_gas_price()
            
            # Build programmable transaction block
            tx_block = {
                "sender": sender_address,
                "gasConfig": {
                    "budget": str(gas_budget),
                    "price": str(gas_price)
                },
                "kind": "programmable",
                "inputs": [
                    {
                        "type": "pure",
                        "valueType": "u64",
                        "value": str(amount_mist)
                    }
                ],
                "transactions": [
                    {
                        "kind": "moveCall",
                        "target": f"{contract}::dex::{function}",
                        "arguments": [
                            {"Input": 0}  # Amount from inputs
                        ]
                    }
                ]
            }
            
            response = await self.network_manager.make_request(
                "sui_buildTransactionBlock",
                [tx_block]
            )
            
            return response['txBytes']
            
        except Exception as e:
            logger.error(f"Failed to build trade transaction: {str(e)}")
            raise

    async def _get_reference_gas_price(self) -> int:
        """Get current gas price"""
        response = await self.network_manager.make_request(
            "suix_getReferenceGasPrice",
            []
        )
        return int(response)
class SuiTransactionExecutor:
    """Handles execution of SUI transactions"""
    
    def __init__(self, network_manager: "NetworkManager"):
        self.network_manager = network_manager
        self.tx_builder = SuiTransactionBuilder(network_manager)

    async def distribute_sui(self, sender_wallet: "SuiWallet", recipient_address: str, 
                           amount_sui: Decimal) -> bool:
        """Distribute SUI to a single recipient"""
        try:
            amount_mist = int(amount_sui * 1000000000)
            
            # Get coins and select appropriate one
            coins = await self._get_coins(sender_wallet.address)
            selected_coin = await self._select_coin(coins, amount_mist)
            
            if not selected_coin:
                logger.error(f"No suitable coin found for distribution of {amount_sui} SUI")
                return False

            # Build transfer transaction
            gas_budget = 2000000  # 2 SUI in MIST
            tx_bytes = await self.tx_builder.build_transfer_tx(
                sender_wallet.address,
                selected_coin["coinObjectId"],
                recipient_address,
                amount_mist,
                gas_budget
            )
            
            # Sign transaction
            signature = sender_wallet.sign_transaction(tx_bytes)
            
            # Execute transaction
            success = await self._execute_transaction(tx_bytes, signature)
            
            if success:
                logger.info(f"Successfully distributed {amount_sui} SUI to {recipient_address}")
            else:
                logger.error(f"Failed to distribute {amount_sui} SUI to {recipient_address}")
                
            return success
            
        except Exception as e:
            logger.error(f"Distribution failed: {str(e)}")
            return False

    async def execute_trade(self, wallet: "SuiWallet", contract: str, 
                          is_buy: bool, amount_sui: Decimal) -> bool:
        """Execute a trade transaction"""
        try:
            amount_mist = int(amount_sui * 1000000000)
            gas_budget = 2000000
            
            # Build trade transaction
            function = "buy" if is_buy else "sell"
            tx_bytes = await self.tx_builder.build_trade_tx(
                wallet.address,
                contract,
                function,
                amount_mist,
                gas_budget
            )
            
            # Sign and execute
            signature = wallet.sign_transaction(tx_bytes)
            success = await self._execute_transaction(tx_bytes, signature)
            
            if success:
                logger.info(f"Successfully executed {function} trade for {amount_sui} SUI")
            else:
                logger.error(f"Failed to execute {function} trade for {amount_sui} SUI")
                
            return success
            
        except Exception as e:
            logger.error(f"Trade execution failed: {str(e)}")
            return False

    async def batch_distribute_sui(self, sender_wallet: "SuiWallet", 
                                 distributions: List[Dict[str, Union[str, Decimal]]]) -> bool:
        """Distribute SUI to multiple recipients"""
        try:
            for dist in distributions:
                success = await self.distribute_sui(
                    sender_wallet,
                    dist['recipient'],
                    dist['amount']
                )
                
                if not success:
                    logger.error(f"Failed to distribute to {dist['recipient']}")
                    return False
                    
                # Add delay between transactions
                await asyncio.sleep(2)
                
            return True
            
        except Exception as e:
            logger.error(f"Batch distribution failed: {str(e)}")
            return False

    async def _get_coins(self, address: str) -> List[Dict]:
        """Get available coins for address"""
        response = await self.network_manager.make_request(
            "suix_getCoins",
            [address, "0x2::sui::SUI", None, 100]
        )
        return response.get("data", [])

    async def _select_coin(self, coins: List[Dict], amount_mist: int) -> Optional[Dict]:
        """Select appropriate coin for transaction"""
        total_needed = amount_mist + 2000000  # Amount + base gas budget
        
        # Sort coins by balance descending
        sorted_coins = sorted(coins, key=lambda x: int(x['balance']), reverse=True)
        
        # Find first coin with sufficient balance
        for coin in sorted_coins:
            if int(coin['balance']) >= total_needed:
                return coin
                
        return None

    async def _execute_transaction(self, tx_bytes: str, signature: List[str]) -> bool:
        """Execute transaction and verify success"""
        try:
            response = await self.network_manager.make_request(
                "sui_executeTransactionBlock",
                [
                    tx_bytes,
                    signature,
                    {
                        "showEffects": True,
                        "showEvents": True,
                        "waitForLocalExecution": True
                    }
                ]
            )
            
            effects = response.get("effects", {})
            status = effects.get("status", {}).get("status")
            
            return status == "success"
            
        except Exception as e:
            logger.error(f"Transaction execution failed: {str(e)}")
            return False

class SuiWallet:
    """Base Sui wallet implementation"""
    
    def __init__(self, private_key: Optional[bytes] = None):
        if private_key is None:
            self.private_key = os.urandom(32)
        else:
            self.private_key = private_key

        self.signing_key = ed25519.SigningKey(self.private_key)
        self.public_key = self.signing_key.get_verifying_key().to_bytes()
        self._address = self._generate_address()

    def _generate_address(self) -> str:
        """Generate Sui address using the official scheme"""
        scheme = bytes([0x00])  # ED25519 scheme identifier
        data = scheme + self.public_key
        address_bytes = hashlib.blake2b(data, digest_size=32).digest()
        return f"0x{address_bytes.hex()}"

    @property
    def address(self) -> str:
        return self._address

    def sign_message(self, message: bytes) -> str:
        """Sign a message using Ed25519"""
        signing_key = ed25519.SigningKey(self.private_key)
        signature = signing_key.sign(message)
        return base64.b64encode(signature).decode()

    def get_public_key_base64(self) -> str:
        """Get base64 encoded public key"""
        return base64.b64encode(self.public_key).decode()

    def get_private_key_hex(self) -> str:
        """Get hex encoded private key"""
        return self.private_key.hex()

class EnhancedWallet(SuiWallet):
    """Enhanced wallet implementation with additional transaction capabilities"""
    
    def sign_transaction(self, tx_bytes_base64: str) -> List[str]:
        """Sign transaction with proper intent and hashing"""
        try:
            # Decode Base64 to raw bytes
            tx_bytes = base64.b64decode(tx_bytes_base64)
            
            # Add intent and create message for signing
            intent_bytes = bytes([0]) + bytes([0]) + bytes([0])  # scope + version + appId
            message_to_sign = intent_bytes + tx_bytes
            
            # Hash the message using Blake2b (32 bytes)
            message_hash = hashlib.blake2b(message_to_sign, digest_size=32).digest()
            
            # Sign the hash
            signature = self.signing_key.sign(message_hash)
            
            # Create final signature bytes with flag
            flag = bytes([0x00])  # ED25519 flag
            final_signature_bytes = flag + signature + self.public_key
            
            # Return Base64 encoded signature
            return [base64.b64encode(final_signature_bytes).decode()]
            
        except Exception as e:
            logger.error(f"Failed to sign transaction: {str(e)}")
            raise

class EndpointManager:
    """Manages multiple RPC endpoints and their usage"""
    
    def __init__(self, network_type: str):
        self.network_type = network_type
        self.endpoints = NETWORKS[network_type]
        self.metrics = MetricsCollector()
        self.active_endpoints = {}
        self.blacklisted = {}
        self.initialize_endpoints()
        
    def initialize_endpoints(self):
        """Initialize all endpoints"""
        for provider, details in self.endpoints.items():
            self.active_endpoints[provider] = {
                **details,
                'health_check_interval': 300,  # 5 minutes
                'last_health_check': 0,
                'consecutive_failures': 0
            }
            
    async def get_endpoint(self) -> Tuple[str, Dict]:
        """Get best available endpoint"""
        available = []
        
        for provider, endpoint in self.active_endpoints.items():
            if provider in self.blacklisted:
                if time.time() - self.blacklisted[provider] > 300:  # 5 minute timeout
                    del self.blacklisted[provider]
                else:
                    continue
                    
            if time.time() - endpoint['last_health_check'] > endpoint['health_check_interval']:
                endpoint['last_health_check'] = time.time()
                
            available.append((provider, endpoint))
                
        if not available:
            raise Exception("No endpoints available")
            
        # Return first available endpoint for now
        return available[0]

    def blacklist_endpoint(self, provider: str):
        """Temporarily blacklist an endpoint"""
        self.blacklisted[provider] = time.time()
        logger.warning(f"Endpoint blacklisted: {provider}")

class NetworkMetrics:
    """Track network performance metrics"""
    request_count: int = 0
    error_count: int = 0
    total_latency: float = 0.0
    last_reset: float = field(default_factory=time.time)
    
    @property
    def average_latency(self) -> float:
        return self.total_latency / self.request_count if self.request_count > 0 else 0
    
    @property
    def error_rate(self) -> float:
        return self.error_count / self.request_count if self.request_count > 0 else 0
    
    def reset(self):
        self.request_count = 0
        self.error_count = 0
        self.total_latency = 0.0
        self.last_reset = time.time()        

class MetricsCollector:
    """Collect and export metrics"""
    def __init__(self, export_path: str = "metrics.json"):
        self.export_path = export_path
        self.metrics: Dict[str, NetworkMetrics] = {}
        self.start_time = time.time()
        
    def record_request(self, provider: str, latency: float, success: bool):
        if provider not in self.metrics:
            self.metrics[provider] = NetworkMetrics()
        
        metrics = self.metrics[provider]
        metrics.request_count += 1
        metrics.total_latency += latency
        if not success:
            metrics.error_count += 1

class NetworkManager:
    """Manages network connections and RPC calls"""
    
    def __init__(self, network_type: str):
        self.endpoint_manager = EndpointManager(network_type)
        self.session = None
        self.headers = {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        # Define retry constants directly in the class
        self.MAX_RETRIES = 3
        self.RETRY_DELAY = 2
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            headers=self.headers,
            timeout=aiohttp.ClientTimeout(total=30)
        )
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
            
    async def make_request(self, method: str, params: List) -> Dict:
        """Make JSON-RPC request with retry logic"""
        retry_count = 0
        
        while retry_count < self.MAX_RETRIES:  # Use self.MAX_RETRIES instead of SuiConstants
            try:
                if not self.session:
                    self.session = aiohttp.ClientSession(
                        timeout=aiohttp.ClientTimeout(total=30)
                    )

                provider, endpoint = await self.endpoint_manager.get_endpoint()
                
                payload = {
                    "jsonrpc": "2.0",
                    "id": str(random.randint(1, 100000)),
                    "method": method,
                    "params": params
                }
                
                async with self.session.post(
                    endpoint['url'],
                    json=payload,
                    headers=endpoint.get('headers', self.headers)
                ) as response:
                    if response.status == 429:  # Rate limit
                        retry_count += 1
                        await asyncio.sleep(self.RETRY_DELAY)
                        continue
                        
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"RPC request failed: {response.status}, {error_text}")
                        
                    result = await response.json()
                    
                    if "error" in result:
                        raise Exception(f"RPC error: {result['error']}")
                        
                    return result.get("result", {})
                    
            except Exception as e:
                retry_count += 1
                if retry_count < self.MAX_RETRIES:
                    await asyncio.sleep(self.RETRY_DELAY * (2 ** retry_count))
                    continue
                raise Exception(f"Request failed after {self.MAX_RETRIES} retries: {str(e)}")

        raise Exception("Max retries reached")

class NetworkHandler:
    """Enhanced network handling with retries and failover"""
    
    def __init__(self, network_manager: NetworkManager):
        self.network_manager = network_manager
        self.MAX_RETRIES = 3
        self.RETRY_DELAY = 2
        self.session = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30)
        )
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
            
    async def execute_with_retry(self, method: str, params: List, 
                               require_success: bool = True) -> Dict:
        """Execute request with retry logic"""
        last_error = None
        
        for attempt in range(self.MAX_RETRIES):
            try:
                result = await self.network_manager.make_request(method, params)
                
                if require_success:
                    if self._verify_response(result):
                        return result
                    raise Exception("Response verification failed")
                return result
                
            except Exception as e:
                last_error = e
                if attempt < self.MAX_RETRIES - 1:
                    await asyncio.sleep(self.RETRY_DELAY * (2 ** attempt))
                    continue
                    
        raise Exception(f"Operation failed after {self.MAX_RETRIES} attempts: {last_error}")
        
    def _verify_response(self, response: Dict) -> bool:
        """Verify response success"""
        if "error" in response:
            return False
            
        effects = response.get("effects", {})
        status = effects.get("status", {}).get("status")
        
        return status == "success"

class WalletManager:
    """Manages wallet operations and transaction signing"""
    
    def __init__(self, network_handler: NetworkHandler):
        self.network_handler = network_handler
        self.wallets: Dict[str, EnhancedWallet] = {}
        
    def create_wallet(self, private_key: Optional[bytes] = None) -> EnhancedWallet:
        """Create new wallet instance"""
        wallet = EnhancedWallet(private_key)
        self.wallets[wallet.address] = wallet
        return wallet
        
    async def verify_wallet_balance(self, wallet: EnhancedWallet, required_amount: Decimal) -> bool:
        """Verify wallet has sufficient balance"""
        try:
            coins = await self.network_handler.execute_with_retry(
                "suix_getCoins",
                [wallet.address, "0x2::sui::SUI", None, 100],
                require_success=False
            )
            
            total_balance = sum(int(coin['balance']) for coin in coins.get("data", []))
            required_mist = int(required_amount * 1000000000)
            
            return total_balance >= required_mist
            
        except Exception as e:
            logger.error(f"Balance verification failed: {str(e)}")
            return False
            
    async def prepare_transaction(self, wallet: EnhancedWallet, 
                                tx_builder: SuiTransactionBuilder,
                                amount: Decimal) -> Optional[str]:
        """Prepare transaction for signing"""
        try:
            # Verify balance
            if not await self.verify_wallet_balance(wallet, amount):
                logger.error(f"Insufficient balance for transaction")
                return None
                
            # Get optimal coin
            coins = await self.network_handler.execute_with_retry(
                "suix_getCoins",
                [wallet.address, "0x2::sui::SUI", None, 100],
                require_success=False
            )
            
            amount_mist = int(amount * 1000000000)
            total_needed = amount_mist + 2000000  # Amount + base gas
            
            selected_coin = next(
                (coin for coin in coins.get("data", []) 
                 if int(coin['balance']) >= total_needed),
                None
            )
            
            if not selected_coin:
                logger.error(f"No suitable coin found")
                return None
                
            return selected_coin['coinObjectId']
            
        except Exception as e:
            logger.error(f"Transaction preparation failed: {str(e)}")
            return None

class UpdatedVolumeMakerBot:
    def __init__(
        self,
        config: BotConfig,
        network_manager: NetworkManager,
        main_wallet: EnhancedWallet,
        trading_wallets: List[EnhancedWallet]
    ):
        self.config = config
        self.network_manager = network_manager
        self.main_wallet = main_wallet
        self.trading_wallets = trading_wallets
        self.is_running = False
        self.is_paused = False
        
        # FlowX specific initialization
        self.flowx_constants = FLOWX_CONSTANTS['TESTNET'] if config.network_type == 'testnet' else FLOWX_CONSTANTS['MAINNET']
        if FLOWX_CONSTANTS['DEBUG_MODE']:
            print(f"\nInitialized with FlowX {config.network_type} configuration")
            print(f"Package ID: {self.flowx_constants['PACKAGE_ID']}")
            print(f"Pool Registry: {self.flowx_constants['POOL_REGISTRY']}")

    # =====================================
    # Original Working Distribution Method - DO NOT MODIFY
    # =====================================
    async def distribute_initial_funds(self, amounts: List[Decimal]) -> bool:
        """Distribute SUI to trading wallets"""
        try:
            for wallet, amount in zip(self.trading_wallets, amounts):
                # Convert SUI to MIST
                amount_mist = int(amount * 1_000_000_000)
                
                # Get coins from main wallet
                response = await self.network_manager.make_request(
                    "suix_getCoins",
                    [self.main_wallet.address, "0x2::sui::SUI", None, 100]
                )
                
                coins = response.get("data", [])
                if not coins:
                    print("No coins found in main wallet")
                    return False

                # Find a coin with sufficient balance
                gas_budget = 2_000_000
                total_needed = amount_mist + gas_budget
                selected_coin = None
                
                for coin in coins:
                    if int(coin['balance']) >= total_needed:
                        selected_coin = coin
                        break

                if not selected_coin:
                    print(f"No coin with sufficient balance. Need {total_needed}")
                    return False

                print(f"Transferring {amount} SUI to {wallet.address}")
                
                # Build Pay transaction
                tx_params = [
                    self.main_wallet.address,
                    [selected_coin["coinObjectId"]],
                    [wallet.address],
                    [str(amount_mist)],
                    str(gas_budget)
                ]

                try:
                    # Get transaction bytes
                    response = await self.network_manager.make_request(
                        "unsafe_paySui",
                        tx_params
                    )
                    
                    tx_bytes = response.get("txBytes")
                    
                    # Sign and execute
                    signature = self.main_wallet.sign_transaction(tx_bytes)
                    
                    execute_response = await self.network_manager.make_request(
                        "sui_executeTransactionBlock",
                        [tx_bytes, signature, {
                            "showEffects": True,
                            "showEvents": True,
                            "waitForLocalExecution": True
                        }]
                    )
                    
                    effects = execute_response.get("effects", {})
                    status = effects.get("status", {}).get("status")
                    
                    if status != "success":
                        print(f"Transaction failed with status: {status}")
                        return False
                        
                    print(f"Successfully transferred {amount} SUI to {wallet.address}")
                    
                except Exception as e:
                    print(f"Transaction failed: {str(e)}")
                    return False

                await asyncio.sleep(2)

            return True
            
        except Exception as e:
            print(f"Distribution error: {str(e)}")
            return False

    # =====================================
    # Updated FlowX Trading Methods
    # =====================================
    async def validate_dex_execution(self, wallet: EnhancedWallet, amount: Decimal) -> bool:
        """Validate before executing DEX trade"""
        try:
            # Check wallet balance
            response = await self.network_manager.make_request(
                "suix_getCoins",
                [wallet.address, "0x2::sui::SUI", None, 100]
            )
            
            coins = response.get("data", [])
            if not coins:
                print(f"No coins found in wallet {wallet.address}")
                return False

            total_balance = sum(int(coin['balance']) for coin in coins)
            amount_mist = int(amount * 1_000_000_000)
            required_balance = amount_mist + 2_000_000  # Amount + gas

            if total_balance < required_balance:
                print(f"Insufficient balance. Required: {required_balance}, Available: {total_balance}")
                return False

            return True

        except Exception as e:
            print(f"Validation error: {str(e)}")
            return False

    async def build_flowx_swap_transaction(self, wallet: EnhancedWallet, is_buy: bool, amount: Decimal) -> Tuple[Optional[str], Optional[str]]:
        """Build FlowX swap transaction"""
        try:
            # Debug Point 1: Transaction Build Start
            print(f"\n=== Building FlowX Transaction ===")
            print(f"Wallet: {wallet.address}")
            print(f"Operation: {'Buy' if is_buy else 'Sell'}")
            print(f"Amount: {amount} SUI")

            amount_mist = int(amount * 1_000_000_000)
            gas_budget = 2_000_000

            # Build transaction
            tx_block = {
                "sender": wallet.address,
                "programmableTransaction": {
                    "inputs": [
                        {
                            "type": "pure",
                            "valueType": "u64",
                            "value": str(amount_mist)
                        }
                    ],
                    "transactions": [
                        {
                            "moveCall": {
                                "target": f"{self.flowx_constants['PACKAGE_ID']}::router::swap_exact_input",
                                "typeArguments": [
                                    "0x2::sui::SUI",
                                    self.config.token_contract
                                ],
                                "arguments": [{"Input": 0}]
                            }
                        }
                    ]
                },
                "gasConfig": {
                    "budget": str(gas_budget),
                    "price": "1000"
                }
            }

            # Debug Point 2: Show Transaction
            print("\nTransaction block being sent:")
            print(json.dumps(tx_block, indent=2))

            # Get transaction bytes
            build_response = await self.network_manager.make_request(
                "sui_buildTransactionBlock",
                [tx_block]
            )

            if "error" in build_response:
                print(f"Transaction building failed: {build_response['error']}")
                return None, None

            return build_response.get("txBytes"), self.flowx_constants['POOL_REGISTRY']

        except Exception as e:
            print(f"Failed to build FlowX swap transaction: {str(e)}")
            return None, None

    async def execute_trade(self, wallet: EnhancedWallet, is_buy: bool, amount: Decimal) -> bool:
        """Execute a token trade using FlowX DEX"""
        try:
            if not await self.validate_dex_execution(wallet, amount):
                return False

            tx_bytes, pool_id = await self.build_flowx_swap_transaction(wallet, is_buy, amount)
            if not tx_bytes or not pool_id:
                return False

            # Sign and execute transaction
            try:
                signature = wallet.sign_transaction(tx_bytes)
                
                execute_response = await self.network_manager.make_request(
                    "sui_executeTransactionBlock",
                    [tx_bytes, signature, {
                        "showEffects": True,
                        "showEvents": True,
                        "waitForLocalExecution": True
                    }]
                )

                effects = execute_response.get("effects", {})
                status = effects.get("status", {}).get("status")

                if status != "success":
                    print(f"Trade failed with status: {status}")
                    return False

                print(f"Successfully executed {'buy' if is_buy else 'sell'} trade")
                print(f"Amount: {amount} SUI")
                return True

            except Exception as e:
                print(f"Trade execution failed: {str(e)}")
                return False

        except Exception as e:
            print(f"Trade error: {str(e)}")
            return False

    async def execute_trade_cycle(self, wallet: EnhancedWallet) -> bool:
        """Execute a trade cycle (random buy/sell)"""
        try:
            # Calculate trade amount with fluctuation
            base_amount = float(self.config.base_amount)
            fluctuation = float(random.uniform(
                -float(self.config.fluctuation_range),
                float(self.config.fluctuation_range)
            ))
            amount = Decimal(str(base_amount + fluctuation))
            
            # Random buy/sell decision
            is_buy = random.random() > 0.5
            
            return await self.execute_trade(wallet, is_buy, amount)
            
        except Exception as e:
            print(f"Trade cycle error: {str(e)}")
            return False

    async def start(self):
        """Start the trading bot"""
        print("\nStarting bot operation...")
        self.is_running = True
        start_time = time.time()
        
        try:
            while self.is_running and \
                  (time.time() - start_time) < (self.config.run_duration * 3600):
                if self.is_paused:
                    await asyncio.sleep(1)
                    continue
                    
                for wallet in self.trading_wallets:
                    if not self.is_running:
                        break
                        
                    try:
                        # Execute trade cycle
                        success = await self.execute_trade_cycle(wallet)
                        if success:
                            print(f"Trade cycle completed successfully for {wallet.address}")
                        else:
                            print(f"Trade cycle failed for {wallet.address}")
                        
                        # Random delay between trades
                        delay = random.randint(
                            self.config.min_interval,
                            self.config.max_interval
                        )
                        print(f"Waiting {delay} seconds before next trade...")
                        await asyncio.sleep(delay)
                        
                    except Exception as e:
                        print(f"Error in trade cycle: {str(e)}")
                        await asyncio.sleep(5)  # Short delay on error
                        continue
                    
        except Exception as e:
            print(f"Bot runtime error: {str(e)}")
        finally:
            self.is_running = False
            print("Bot stopped")

    def stop(self):
        """Stop the bot"""
        self.is_running = False
        print("Stopping bot...")

    def pause(self):
        """Pause the bot"""
        self.is_paused = True
        print("Bot paused")

    def resume(self):
        """Resume the bot"""
        self.is_paused = False
        print("Bot resumed")

# Rest of your bot implementation methods...
# Updated main execution
# This should be at the very end of your file, after all class definitions

def validate_custom_slippage(min_slippage: Decimal, max_slippage: Decimal) -> bool:
    """Validate custom slippage inputs"""
    try:
        # Check minimum bounds
        if min_slippage < MIN_ALLOWED_SLIPPAGE:
            print(f"Minimum slippage cannot be less than {MIN_ALLOWED_SLIPPAGE}%")
            return False
            
        # Check maximum bounds
        if max_slippage > MAX_ALLOWED_SLIPPAGE:
            print(f"Maximum slippage cannot exceed {MAX_ALLOWED_SLIPPAGE}%")
            return False
            
        # Check relationship
        if min_slippage >= max_slippage:
            print("Minimum slippage must be less than maximum slippage")
            return False
            
        # Warning for high slippage
        if max_slippage > HIGH_SLIPPAGE_WARNING:
            print(f"\nWARNING: High slippage detected ({max_slippage}%)")
            print("Higher slippage increases the risk of unfavorable trades")
            
        # Extra warning for extreme slippage
        if max_slippage > EXTREME_SLIPPAGE_WARNING:
            print("\nCAUTION: Extremely high slippage detected!")
            print("This significantly increases your risk of unfavorable trades")
            confirm = input("Are you sure you want to proceed? (yes/no): ")
            return confirm.lower() == 'yes'
            
        return True
        
    except Exception as e:
        print(f"Error validating slippage: {str(e)}")
        return False

async def main():
    try:
        print("\n=== SUI Volume Maker Bot v2 ===\n")

        # Network selection
        network_type = input("Select network (mainnet/testnet): ").strip().lower()
        if network_type not in ["mainnet", "testnet"]:
            print("Invalid network selection")
            return

        # Initialize network manager
        network_manager = NetworkManager(network_type)
        
        async with network_manager as nm:
            try:
                # Create enhanced wallets with private key display
                print("\nCreating main wallet...")
                main_wallet = EnhancedWallet()
                print(f"Main wallet address: {main_wallet.address}")
                print(f"Main wallet private key: {main_wallet.get_private_key_hex()}")
                print("-" * 50)
                
                num_wallets = int(input("\nNumber of trading wallets to create: "))
                
                # Check main wallet balance
                print("\nChecking main wallet balance...")
                try:
                    # Simple balance check
                    response = await nm.make_request(
                        "suix_getCoins",
                        [main_wallet.address, "0x2::sui::SUI", None, 100]
                    )
                    
                    coins = response.get("data", [])
                    total_balance = sum(int(coin['balance']) for coin in coins)
                    balance_sui = Decimal(total_balance) / Decimal(1_000_000_000)
                    
                    print(f"\nMain wallet balance: {balance_sui} SUI")
                    
                    while balance_sui == 0:
                        print("\nMain wallet needs funding before proceeding.")
                        if input("Check balance again? (yes/no): ").lower() != 'yes':
                            return
                            
                        response = await nm.make_request(
                            "suix_getCoins",
                            [main_wallet.address, "0x2::sui::SUI", None, 100]
                        )
                        coins = response.get("data", [])
                        total_balance = sum(int(coin['balance']) for coin in coins)
                        balance_sui = Decimal(total_balance) / Decimal(1_000_000_000)
                        print(f"Updated balance: {balance_sui} SUI")
                        
                except Exception as e:
                    print(f"Error checking balance: {e}")
                    return
                
                # Create trading wallets
                trading_wallets = []
                print("\nCreating trading wallets...")
                for i in range(num_wallets):
                    wallet = EnhancedWallet()
                    trading_wallets.append(wallet)
                    print(f"\nWallet {i+1}:")
                    print(f"Address: {wallet.address}")
                    print(f"Private Key: {wallet.get_private_key_hex()}")
                    print("-" * 50)

                # Save wallet details to file
                with open('wallet_details.txt', 'w') as f:
                    f.write("=== Wallet Details ===\n\n")
                    f.write("Main Wallet:\n")
                    f.write(f"Address: {main_wallet.address}\n")
                    f.write(f"Private Key: {main_wallet.get_private_key_hex()}\n")
                    f.write("-" * 50 + "\n\n")
                    
                    f.write("Trading Wallets:\n")
                    for i, wallet in enumerate(trading_wallets, 1):
                        f.write(f"\nWallet {i}:\n")
                        f.write(f"Address: {wallet.address}\n")
                        f.write(f"Private Key: {wallet.get_private_key_hex()}\n")
                        f.write("-" * 50 + "\n")
                
                print("\nWallet details have been saved to 'wallet_details.txt'")

                # Get distribution amounts
                print("\nSpecify distribution amounts:")
                distribution_amounts = []
                total_needed = Decimal('0')
                
                for i, wallet in enumerate(trading_wallets, 1):
                    while True:
                        try:
                            amount = Decimal(input(f"Amount for wallet {i} ({wallet.address}): "))
                            if amount > 0:
                                distribution_amounts.append(amount)
                                total_needed += amount
                                break
                        except:
                            print("Please enter a valid amount")

                # Verify sufficient balance for distribution
                total_needed_mist = int(total_needed * 1000000000)
                gas_budget = 2000000  # 2 SUI in MIST per transaction
                total_required = total_needed_mist + (gas_budget * num_wallets)
                
                if total_balance < total_required:
                    print("\nInsufficient balance in main wallet")
                    print(f"Required: {Decimal(total_required) / Decimal(1000000000)} SUI")
                    print(f"Available: {balance_sui} SUI")
                    return

                # In your main function, replace the part after balance verification with this:

                print("\nBalance verification successful")
                print(f"Required: {Decimal(total_required) / Decimal(1000000000)} SUI")
                print(f"Available: {balance_sui} SUI")

                # Initialize temporary bot for distribution
                temp_config = BotConfig(
                    network_type=network_type,
                    token_contract="",  # Temporary empty value
                    package_id="",      # Temporary empty value
                    base_amount=Decimal('0'),
                    fluctuation_range=Decimal('0'),
                    min_interval=0,
                    max_interval=0,
                    run_duration=0,
                    fee_percentage=Decimal('0'),
                    num_wallets=num_wallets
                )
                
                temp_bot = UpdatedVolumeMakerBot(temp_config, nm, main_wallet, trading_wallets)

                # Distribute initial funds
                print("\nDistributing initial funds...")
                if not await temp_bot.distribute_initial_funds(distribution_amounts):
                    print("Failed to distribute funds. Exiting...")
                    return

                # Verify the distributions
                print("\nVerifying distributions...")
                for wallet, amount in zip(trading_wallets, distribution_amounts):
                    try:
                        response = await nm.make_request(
                            "suix_getCoins",
                            [wallet.address, "0x2::sui::SUI", None, 100]
                        )
                        coins = response.get("data", [])
                        wallet_balance = sum(int(coin['balance']) for coin in coins)
                        wallet_balance_sui = Decimal(wallet_balance) / Decimal(1000000000)
                        
                        if wallet_balance_sui < amount:
                            print(f"\nDistribution verification failed for wallet {wallet.address}")
                            print(f"Expected: {amount} SUI")
                            print(f"Received: {wallet_balance_sui} SUI")
                            return
                            
                        print(f"Verified wallet {wallet.address}: {wallet_balance_sui} SUI")
                    except Exception as e:
                        print(f"Error verifying distribution: {e}")
                        return

                print("\nAll distributions completed and verified successfully!")
                
                # Proceed with trading configuration
                print("\nAll distributions completed and verified successfully!")
                print("\n=== Trading Configuration ===")
                input("\nPress Enter to continue with trading configuration...")
                
                # Now proceed with trading configuration
                config = BotConfig(
                    network_type=network_type,
                    token_contract=input("Token contract address: ").strip(),
                    package_id=input("Package ID: ").strip(),
                    base_amount=Decimal(input("Base transaction amount: ")),
                    fluctuation_range=Decimal(input("Fluctuation range: ")),
                    min_interval=int(input("Minimum interval (seconds): ")),
                    max_interval=int(input("Maximum interval (seconds): ")),
                    run_duration=int(input("Run duration (hours): ")),
                    fee_percentage=Decimal(input("Fee percentage: ")),
                    num_wallets=num_wallets
                )

                # Initialize bot for trading
                bot = UpdatedVolumeMakerBot(config, nm, main_wallet, trading_wallets)
                
                print("\nStarting bot operation...")
                await bot.start()  # Directly start trading without another distribution

            except KeyboardInterrupt:
                print("\nBot stopped by user")
            except Exception as e:
                print(f"Error: {str(e)}")

    except Exception as e:
        print(f"Startup error: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())