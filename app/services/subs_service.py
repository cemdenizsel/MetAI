from typing import Optional
from bson import ObjectId
from data.mongodb_config import get_database


async def create_subscription_record(subscription):
    """Create subscription record in database"""
    try:
        db = await get_database()
        subscriptions_collection = db["subscriptions"]
        
        # Get user_id from metadata
        metadata = subscription.get('metadata', {})
        user_id = metadata.get('user_id')
        
        sub_data = {
            "stripe_subscription_id": subscription['id'],
            "customer_id": subscription['customer'],
            "user_id": ObjectId(user_id) if user_id else None,
            "user_email": metadata.get('user_email'),
            "status": subscription['status'],
            "current_period_start": subscription['current_period_start'],
            "current_period_end": subscription['current_period_end'],
            "created_at": subscription['created'],
            "cancelled_at": None,
            "cancel_at_period_end": False
        }
        
        result = await subscriptions_collection.insert_one(sub_data)
        return result.inserted_id is not None
        
    except Exception as e:
        print(f"Error creating subscription record: {e}")
        return False


async def update_subscription_record(subscription):
    """Update subscription record in database"""
    try:
        db = await get_database()
        subscriptions_collection = db["subscriptions"]
        
        update_data = {
            "status": subscription['status'],
            "current_period_start": subscription['current_period_start'],
            "current_period_end": subscription['current_period_end'],
            "cancel_at_period_end": subscription['cancel_at_period_end']
        }
        
        if subscription.get('canceled_at'):
            update_data["cancelled_at"] = subscription['canceled_at']
        
        result = await subscriptions_collection.update_one(
            {"stripe_subscription_id": subscription['id']},
            {"$set": update_data}
        )
        return result.modified_count > 0
        
    except Exception as e:
        print(f"Error updating subscription record: {e}")
        return False


async def cancel_subscription_record(subscription):
    """Mark subscription as cancelled in database"""
    try:
        db = await get_database()
        subscriptions_collection = db["subscriptions"]
        
        result = await subscriptions_collection.update_one(
            {"stripe_subscription_id": subscription['id']},
            {"$set": {
                "status": "canceled",
                "cancelled_at": subscription.get('canceled_at', subscription.get('ended_at'))
            }}
        )
        return result.modified_count > 0
        
    except Exception as e:
        print(f"Error cancelling subscription record: {e}")
        return False


async def save_successful_invoice(stripe_invoice):
    """Save successful payment invoice to database"""
    try:
        db = await get_database()
        invoices_collection = db["invoices"]
        subscriptions_collection = db["subscriptions"]
        
        # Check if invoice already exists
        existing = await invoices_collection.find_one({"payment_token": stripe_invoice['id']})
        if existing:
            return True
        
        # Get user_id from subscription record
        subscription_doc = await subscriptions_collection.find_one({
            "stripe_subscription_id": stripe_invoice.get('subscription')
        })
        
        user_id = subscription_doc.get('user_id') if subscription_doc else None
        
        invoice_data = {
            "id": ObjectId(),
            "user_id": user_id,
            "is_paid": True,
            "payment_token": stripe_invoice['id'],
            "provider": "Stripe",
            "amount": stripe_invoice['amount_paid'],
            "currency": stripe_invoice['currency'],
            "subscription_id": stripe_invoice.get('subscription'),
            "billing_period_start": stripe_invoice.get('period_start'),
            "billing_period_end": stripe_invoice.get('period_end'),
            "created_at": stripe_invoice['created']
        }
        
        result = await invoices_collection.insert_one(invoice_data)
        return result.inserted_id is not None
        
    except Exception as e:
        print(f"Error saving successful invoice: {e}")
        return False


async def save_failed_invoice(stripe_invoice):
    """Save failed payment invoice to database"""
    try:
        db = await get_database()
        invoices_collection = db["invoices"]
        subscriptions_collection = db["subscriptions"]
        
        # Get user_id from subscription record
        subscription_doc = await subscriptions_collection.find_one({
            "stripe_subscription_id": stripe_invoice.get('subscription')
        })
        
        user_id = subscription_doc.get('user_id') if subscription_doc else None
        
        invoice_data = {
            "id": ObjectId(),
            "user_id": user_id,
            "is_paid": False,
            "payment_token": stripe_invoice['id'],
            "provider": "Stripe",
            "amount": stripe_invoice['amount_due'],
            "currency": stripe_invoice['currency'],
            "subscription_id": stripe_invoice.get('subscription'),
            "failed_at": stripe_invoice['created']
        }
        
        result = await invoices_collection.insert_one(invoice_data)
        return result.inserted_id is not None
        
    except Exception as e:
        print(f"Error saving failed invoice: {e}")
        return False


async def get_user_subscription(user_id: str) -> Optional[dict]:
    """Get active subscription for a user"""
    try:
        db = await get_database()
        subscriptions_collection = db["subscriptions"]
        
        subscription = await subscriptions_collection.find_one({
            "user_id": ObjectId(user_id),
            "status": {"$in": ["active", "trialing"]}
        })
        
        return subscription
        
    except Exception as e:
        print(f"Error getting user subscription: {e}")
        return None


async def get_user_invoices(user_id: str) -> list:
    """Get all invoices for a user"""
    try:
        db = await get_database()
        invoices_collection = db["invoices"]
        
        cursor = invoices_collection.find({
            "user_id": ObjectId(user_id)
        }).sort("created_at", -1)
        
        invoices = []
        async for invoice in cursor:
            invoices.append(invoice)
            
        return invoices
        
    except Exception as e:
        print(f"Error getting user invoices: {e}")
        return []


async def check_batch_quota(user_id: str, video_count: int) -> dict:
    """
    Check if user has enough quota for batch processing.
    
    Args:
        user_id: User ID
        video_count: Number of videos to process
        
    Returns:
        Dictionary with quota check result
    """
    try:
        db = get_database()
        quotas_collection = db["user_quotas"]
        
        # Get or create quota record
        user_object_id = ObjectId(user_id)
        quota_doc = quotas_collection.find_one({"user_id": user_object_id})
        
        # Default quota (can be customized per subscription plan)
        default_monthly_quota = 100
        
        if not quota_doc:
            # Create new quota record
            quota_doc = {
                "user_id": user_object_id,
                "monthly_quota": default_monthly_quota,
                "used_this_month": 0,
                "reset_date": None
            }
            quotas_collection.insert_one(quota_doc)
        
        used = quota_doc.get("used_this_month", 0)
        quota = quota_doc.get("monthly_quota", default_monthly_quota)
        remaining = quota - used
        
        if remaining < video_count:
            return {
                "allowed": False,
                "quota": quota,
                "used": used,
                "remaining": remaining,
                "requested": video_count,
                "message": f"Insufficient quota. Requested {video_count} but only {remaining} remaining."
            }
        
        return {
            "allowed": True,
            "quota": quota,
            "used": used,
            "remaining": remaining,
            "requested": video_count
        }
        
    except Exception as e:
        print(f"Error checking batch quota: {e}")
        # Default to allow on error (can be changed to deny for stricter control)
        return {
            "allowed": True,
            "error": str(e)
        }


async def deduct_quota(user_id: str, video_count: int) -> bool:
    """
    Deduct videos from user's quota.
    
    Args:
        user_id: User ID
        video_count: Number of videos processed
        
    Returns:
        True if successful, False otherwise
    """
    try:
        db = get_database()
        quotas_collection = db["user_quotas"]
        
        user_object_id = ObjectId(user_id)
        
        result = quotas_collection.update_one(
            {"user_id": user_object_id},
            {"$inc": {"used_this_month": video_count}}
        )
        
        return result.modified_count > 0
        
    except Exception as e:
        print(f"Error deducting quota: {e}")
        return False


async def get_remaining_quota(user_id: str) -> int:
    """
    Get remaining quota for user.
    
    Args:
        user_id: User ID
        
    Returns:
        Remaining quota count
    """
    try:
        db = get_database()
        quotas_collection = db["user_quotas"]
        
        user_object_id = ObjectId(user_id)
        quota_doc = quotas_collection.find_one({"user_id": user_object_id})
        
        if not quota_doc:
            return 100  # Default quota
        
        used = quota_doc.get("used_this_month", 0)
        quota = quota_doc.get("monthly_quota", 100)
        
        return max(0, quota - used)
        
    except Exception as e:
        print(f"Error getting remaining quota: {e}")
        return 0