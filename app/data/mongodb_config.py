"""
MongoDB Configuration and Connection Management
"""

import os
import logging
from typing import Optional
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import MongoClient
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class MongoDBConfig:
    """MongoDB configuration and connection management."""
    
    def __init__(self):
        self.connection_string = os.getenv(
            "MONGODB_CONNECTION_STRING", 
            "mongodb://localhost:27017"
        )
        self.database_name = os.getenv("MONGODB_DATABASE", "emotion_analysis_db")
        self.chat_collection = os.getenv("MONGODB_CHAT_COLLECTION", "chat_sessions")
        self.users_collection = os.getenv("MONGODB_USERS_COLLECTION", "users")
        
        self.client: Optional[AsyncIOMotorClient] = None
        self.database = None
        self.chat_sessions_collection = None
        self.users_collection_instance = None
        
        logger.info(f"MongoDB config initialized - DB: {self.database_name}")
    
    async def connect(self):
        """Connect to MongoDB."""
        try:
            self.client = AsyncIOMotorClient(self.connection_string)
            
            # Test connection
            await self.client.admin.command('ping')
            
            self.database = self.client[self.database_name]
            self.chat_sessions_collection = self.database[self.chat_collection]
            self.users_collection_instance = self.database[self.users_collection]
            
            # Create indexes for better performance
            await self._create_indexes()
            
            logger.info("Successfully connected to MongoDB")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            self.client = None
            return False
    
    async def disconnect(self):
        """Disconnect from MongoDB."""
        if self.client:
            self.client.close()
            self.client = None
            logger.info("Disconnected from MongoDB")
    
    async def _create_indexes(self):
        """Create necessary indexes for optimal performance."""
        try:
            # Chat sessions indexes
            if self.chat_sessions_collection:
                await self.chat_sessions_collection.create_index("chat_id", unique=True)
                await self.chat_sessions_collection.create_index("created_at")
                await self.chat_sessions_collection.create_index("updated_at")
            
            # Users indexes
            if self.users_collection_instance:
                await self.users_collection_instance.create_index("email", unique=True)
                await self.users_collection_instance.create_index("username", unique=True)
                await self.users_collection_instance.create_index("created_at")
            
            logger.info("MongoDB indexes created successfully")
            
        except Exception as e:
            logger.warning(f"Failed to create indexes: {e}")
    
    def is_connected(self) -> bool:
        """Check if connected to MongoDB."""
        return self.client is not None
    
    async def health_check(self) -> dict:
        """Perform MongoDB health check."""
        if not self.client:
            return {
                "status": "disconnected",
                "error": "No MongoDB connection"
            }
        
        try:
            # Ping database
            await self.client.admin.command('ping')
            
            # Get database stats
            stats = await self.database.command("dbStats")
            
            # Count chat sessions
            chat_count = await self.chat_sessions_collection.count_documents({})
            
            return {
                "status": "healthy",
                "database": self.database_name,
                "chat_sessions_count": chat_count,
                "db_size_mb": round(stats.get("dataSize", 0) / (1024 * 1024), 2),
                "collections": stats.get("collections", 0)
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }


# Global MongoDB configuration instance
mongo_config = MongoDBConfig()


async def get_mongo_client() -> Optional[AsyncIOMotorClient]:
    """Get MongoDB client instance."""
    if not mongo_config.is_connected():
        await mongo_config.connect()
    
    return mongo_config.client


async def get_database():
    """Get MongoDB database instance."""
    if not mongo_config.is_connected():
        await mongo_config.connect()
    
    return mongo_config.database


async def get_chat_collection():
    """Get chat sessions collection."""
    if not mongo_config.is_connected():
        await mongo_config.connect()
    
    return mongo_config.chat_sessions_collection


async def get_users_collection():
    """Get users collection."""
    if not mongo_config.is_connected():
        await mongo_config.connect()
    
    return mongo_config.users_collection_instance


# Sync version for non-async usage
def get_sync_mongo_client() -> Optional[MongoClient]:
    """Get synchronous MongoDB client."""
    try:
        client = MongoClient(mongo_config.connection_string)
        client.admin.command('ping')  # Test connection
        return client
    except Exception as e:
        logger.error(f"Failed to create sync MongoDB client: {e}")
        return None