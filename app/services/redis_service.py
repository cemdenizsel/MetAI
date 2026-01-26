"""
Redis Service for OTP and Session Management

Handles Redis operations for OTP codes, session management, and caching.
"""

import os
from typing import Optional
import redis.asyncio as aioredis
from dotenv import load_dotenv

load_dotenv()


class RedisService:
    """Service class for Redis operations."""
    
    def __init__(self):
        self.redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        self.redis_db = int(os.getenv("REDIS_DB", "0"))
        self.redis_password = os.getenv("REDIS_PASSWORD", None)
        self.pool = None
        self.redis = None
    
    async def connect(self):
        """Connect to Redis."""
        try:
            if self.redis_password:
                self.redis = await aioredis.from_url(
                    self.redis_url,
                    password=self.redis_password,
                    db=self.redis_db,
                    encoding="utf-8",
                    decode_responses=True
                )
            else:
                self.redis = await aioredis.from_url(
                    self.redis_url,
                    db=self.redis_db,
                    encoding="utf-8",
                    decode_responses=True
                )
            
            # Test connection
            await self.redis.ping()
            print("Connected to Redis successfully")
            return True
            
        except Exception as e:
            print(f"Failed to connect to Redis: {e}")
            return False
    
    async def disconnect(self):
        """Disconnect from Redis."""
        if self.redis:
            await self.redis.close()
            print("Disconnected from Redis")
    
    async def set_otp(self, user_email: str, otp_code: str, expire_minutes: int = 10) -> bool:
        """
        Store OTP code for user with expiration.
        
        Args:
            user_email: User's email address
            otp_code: Generated OTP code
            expire_minutes: Expiration time in minutes
            
        Returns:
            True if stored successfully, False otherwise
        """
        try:
            if not self.redis:
                await self.connect()
            
            key = f"otp:{user_email}"
            expire_seconds = expire_minutes * 60
            
            await self.redis.setex(key, expire_seconds, otp_code)
            return True
            
        except Exception as e:
            print(f"Error storing OTP: {e}")
            return False
    
    async def get_otp(self, user_email: str) -> Optional[str]:
        """
        Retrieve OTP code for user.
        
        Args:
            user_email: User's email address
            
        Returns:
            OTP code if exists, None otherwise
        """
        try:
            if not self.redis:
                await self.connect()
            
            key = f"otp:{user_email}"
            otp_code = await self.redis.get(key)
            return otp_code
            
        except Exception as e:
            print(f"Error retrieving OTP: {e}")
            return None
    
    async def delete_otp(self, user_email: str) -> bool:
        """
        Delete OTP code for user (invalidate).
        
        Args:
            user_email: User's email address
            
        Returns:
            True if deleted successfully, False otherwise
        """
        try:
            if not self.redis:
                await self.connect()
            
            key = f"otp:{user_email}"
            result = await self.redis.delete(key)
            return result > 0
            
        except Exception as e:
            print(f"Error deleting OTP: {e}")
            return False
    
    async def verify_otp(self, user_email: str, provided_otp: str) -> bool:
        """
        Verify OTP code and delete it if valid.
        
        Args:
            user_email: User's email address
            provided_otp: OTP code provided by user
            
        Returns:
            True if OTP is valid, False otherwise
        """
        try:
            stored_otp = await self.get_otp(user_email)
            
            if stored_otp and stored_otp == provided_otp:
                # OTP is valid, delete it to prevent reuse
                await self.delete_otp(user_email)
                return True
            
            return False
            
        except Exception as e:
            print(f"Error verifying OTP: {e}")
            return False
    
    async def set_cache(self, key: str, value: str, expire_minutes: int = 60) -> bool:
        """Set cache value with expiration."""
        try:
            if not self.redis:
                await self.connect()
            
            expire_seconds = expire_minutes * 60
            await self.redis.setex(key, expire_seconds, value)
            return True
            
        except Exception as e:
            print(f"Error setting cache: {e}")
            return False
    
    async def get_cache(self, key: str) -> Optional[str]:
        """Get cache value."""
        try:
            if not self.redis:
                await self.connect()
            
            return await self.redis.get(key)
            
        except Exception as e:
            print(f"Error getting cache: {e}")
            return None
    
    async def delete_cache(self, key: str) -> bool:
        """Delete cache value."""
        try:
            if not self.redis:
                await self.connect()
            
            result = await self.redis.delete(key)
            return result > 0
            
        except Exception as e:
            print(f"Error deleting cache: {e}")
            return False
    
    async def set_json(self, key: str, data: dict, expire_seconds: int) -> bool:
        """
        Store JSON data_model with gzip compression.
        
        Args:
            key: Cache key
            data: Data dictionary to store
            expire_seconds: Expiration time in seconds
            
        Returns:
            True if stored successfully, False otherwise
        """
        try:
            if not self.redis:
                await self.connect()
            
            # Serialize to JSON
            import json
            import gzip
            json_str = json.dumps(data)
            
            # Compress with gzip
            compressed_data = gzip.compress(json_str.encode('utf-8'))
            
            # Store in Redis
            await self.redis.setex(key, expire_seconds, compressed_data)
            return True
            
        except Exception as e:
            print(f"Error storing JSON data_model: {e}")
            return False
    
    async def get_json(self, key: str) -> dict:
        """
        Retrieve and decompress JSON data_model.
        
        Args:
            key: Cache key
            
        Returns:
            Data dictionary if exists, None otherwise
        """
        try:
            if not self.redis:
                await self.connect()
            
            # Get compressed data_model
            compressed_data = await self.redis.get(key)
            
            if not compressed_data:
                return None
            
            # Decompress
            import json
            import gzip
            
            # Handle both bytes and string
            if isinstance(compressed_data, str):
                compressed_data = compressed_data.encode('utf-8')
            
            decompressed_data = gzip.decompress(compressed_data)
            json_str = decompressed_data.decode('utf-8')
            
            # Parse JSON
            data = json.loads(json_str)
            return data
            
        except Exception as e:
            print(f"Error retrieving JSON data_model: {e}")
            return None
    
    async def get_cache_size(self) -> dict:
        """
        Get total cache size information.
        
        Returns:
            Cache size information dictionary
        """
        try:
            if not self.redis:
                await self.connect()
            
            # Get Redis info
            info = await self.redis.info()
            
            used_memory = info.get("used_memory", 0)
            used_memory_human = info.get("used_memory_human", "0B")
            
            # Get database size
            db_size = await self.redis.dbsize()
            
            return {
                "used_memory_bytes": used_memory,
                "used_memory_human": used_memory_human,
                "total_keys": db_size
            }
            
        except Exception as e:
            return {
                "error": str(e)
            }
    
    async def get_cache_keys(self, pattern: str) -> list:
        """
        List cache keys matching pattern.
        
        Args:
            pattern: Key pattern (e.g., "emotion_result:*")
            
        Returns:
            List of matching keys
        """
        try:
            if not self.redis:
                await self.connect()
            
            # Scan for keys (more efficient than KEYS command)
            keys = []
            cursor = 0
            
            while True:
                cursor, partial_keys = await self.redis.scan(
                    cursor=cursor,
                    match=pattern,
                    count=100
                )
                keys.extend(partial_keys)
                
                if cursor == 0:
                    break
            
            return keys
            
        except Exception as e:
            print(f"Error getting cache keys: {e}")
            return []
    
    async def health_check(self) -> dict:
        """Perform Redis health check."""
        try:
            if not self.redis:
                await self.connect()
            
            # Ping Redis
            pong = await self.redis.ping()
            
            # Get Redis info
            info = await self.redis.info()
            
            return {
                "status": "healthy" if pong else "unhealthy",
                "ping": pong,
                "version": info.get("redis_version", "unknown"),
                "connected_clients": info.get("connected_clients", 0),
                "used_memory": info.get("used_memory_human", "unknown")
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }


# Global Redis service instance
redis_service = RedisService()


async def get_redis_service() -> RedisService:
    """Get Redis service instance."""
    if not redis_service.redis:
        await redis_service.connect()
    return redis_service