"""
User Service

Handles all user database operations and business logic.
"""

import time
from typing import Optional
from motor.motor_asyncio import AsyncIOMotorCollection

from models.user_models import UserModel
from utils.auth import get_password_hash, verify_password
from data.mongodb_config import get_database


class UserService:
    """Service class for user operations."""
    
    def __init__(self, collection: AsyncIOMotorCollection):
        self.collection = collection
    
    async def create_user(self, username: str, email: str, password: str, package: str = "Basic") -> bool:
        """Create a new user."""
        try:
            # Hash password
            password_hash = get_password_hash(password)
            
            user_doc = {
                "username": username,
                "email": email,
                "password_hash": password_hash,
                "package": package,
                "created_at": time.time(),
                "updated_at": time.time(),
                "is_verified": False,
            }
            
            result = await self.collection.insert_one(user_doc)
            return result.inserted_id is not None
        except Exception as e:
            print(f"Error creating user: {e}")
            return False
    
    async def get_user_by_email(self, email: str) -> Optional[dict]:
        """Get user document by email."""
        try:
            doc = await self.collection.find_one({"email": email})
            return doc
        except Exception as e:
            print(f"Error getting user by email: {e}")
            return None
    
    async def get_user_by_username(self, username: str) -> Optional[dict]:
        """Get user document by username."""
        try:
            doc = await self.collection.find_one({"username": username})
            return doc
        except Exception as e:
            print(f"Error getting user by username: {e}")
            return None
    
    async def email_exists(self, email: str) -> bool:
        """Check if email already exists."""
        try:
            count = await self.collection.count_documents({"email": email})
            return count > 0
        except Exception as e:
            print(f"Error checking email existence: {e}")
            return False
    
    async def username_exists(self, username: str) -> bool:
        """Check if username already exists."""
        try:
            count = await self.collection.count_documents({"username": username})
            return count > 0
        except Exception as e:
            print(f"Error checking username existence: {e}")
            return False
    
    async def authenticate_user(self, email: str, password: str) -> Optional[dict]:
        """Authenticate user with email and password."""
        try:
            user_doc = await self.get_user_by_email(email)
            if not user_doc:
                return None
            
            # Verify password
            if not verify_password(password, user_doc["password_hash"]):
                return None
            
            return user_doc
        except Exception as e:
            print(f"Error authenticating user: {e}")
            return None
    
    async def update_user(self, email: str, update_data: dict) -> bool:
        """Update user data_model."""
        try:
            update_data["updated_at"] = time.time()
            result = await self.collection.update_one(
                {"email": email},
                {"$set": update_data}
            )
            return result.modified_count > 0
        except Exception as e:
            print(f"Error updating user: {e}")
            return False
    
    async def update_password(self, email: str, current_password: str, new_password: str) -> bool:
        """Update user password after verifying current password."""
        try:
            # Get user and verify current password
            user_doc = await self.get_user_by_email(email)
            if not user_doc:
                return False
            
            if not verify_password(current_password, user_doc["password_hash"]):
                return False
            
            # Hash new password and update
            new_password_hash = get_password_hash(new_password)
            return await self.update_user(email, {"password_hash": new_password_hash})
        except Exception as e:
            print(f"Error updating password: {e}")
            return False
    
    async def delete_user(self, email: str) -> bool:
        """Delete a user."""
        try:
            result = await self.collection.delete_one({"email": email})
            return result.deleted_count > 0
        except Exception as e:
            print(f"Error deleting user: {e}")
            return False
    
    async def get_user_count(self) -> int:
        """Get total number of users."""
        try:
            return await self.collection.count_documents({})
        except Exception as e:
            print(f"Error counting users: {e}")
            return 0
    
    def doc_to_user_model(self, user_doc: dict) -> UserModel:
        """Convert MongoDB document to UserModel."""
        return UserModel(
            username=user_doc["username"],
            email=user_doc["email"],
            package=user_doc["package"],
            password=""  # Don't expose password
        )
    
    def doc_to_api_response(self, user_doc: dict) -> dict:
        """Convert MongoDB document to API response format."""
        return {
            "username": user_doc["username"],
            "email": user_doc["email"],
            "package": user_doc["package"],
            "created_at": user_doc["created_at"],
            "updated_at": user_doc["updated_at"]
        }


# Dependency to get user service
async def get_user_service() -> UserService:
    """Get user service instance."""
    db = await get_database()
    collection = db.users
    return UserService(collection)