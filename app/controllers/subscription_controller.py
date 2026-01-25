import os

import stripe
from fastapi import APIRouter, Request, HTTPException, Depends
from fastapi.responses import RedirectResponse
from models.subs_model import CheckoutRequest
from utils.auth import get_current_user_email
from services.user_service import get_user_service, UserService
from services.subs_service import (
    create_subscription_record,
    update_subscription_record,
    cancel_subscription_record,
    save_successful_invoice,
    save_failed_invoice,
    get_user_subscription,
    get_user_invoices
)

router = APIRouter(prefix="/subs", tags=["subscription"])
stripe.api_key = os.getenv("STRIPE_SECRET_KEY")


@router.post("/checkout")
async def create_checkout(
    request: CheckoutRequest,
    current_user_email: str = Depends(get_current_user_email),
    user_service: UserService = Depends(get_user_service)
):
    domain = os.getenv("DOMAIN_NAME")
    try:
        # Get user info from token
        user_doc = await user_service.get_user_by_email(current_user_email)
        if not user_doc:
            raise HTTPException(status_code=404, detail="User not found")
        
        prices = stripe.Price.list(
            lookup_keys=[request.lookup_key],
            expand=['data_model.product']
        )

        checkout_session = stripe.checkout.Session.create(
            line_items=[
                {
                    'price': prices.data[0].id,
                    'quantity': 1,
                },
            ],
            mode='subscription',
            success_url=domain + '?success=true&session_id={CHECKOUT_SESSION_ID}',
            cancel_url=domain + '?canceled=true',
            customer_email=current_user_email,
            subscription_data={
                'trial_period_days': 7,
                'metadata': {
                    'lookup_key': request.lookup_key,
                    'user_id': str(user_doc['_id']),
                    'user_email': current_user_email
                }
            }
        )
        return RedirectResponse(checkout_session.url, status_code=303)
    except Exception as e:
        print(e)
        return "Server error", 500


@router.post("/webhook")
async def stripe_webhook(request: Request):
    payload = await request.body()
    sig_header = request.headers.get('stripe-signature')
    endpoint_secret = os.getenv("STRIPE_WEBHOOK_SECRET")
    
    try:
        event = stripe.Webhook.construct_event(
            payload, sig_header, endpoint_secret
        )
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid payload")
    except stripe.error.SignatureVerificationError:
        raise HTTPException(status_code=400, detail="Invalid signature")
    
    # Handle subscription events
    if event['type'] == 'customer.subscription.created':
        success = await create_subscription_record(event['data_model']['object'])
        print(f"Subscription created: {event['data_model']['object']['id']}" if success else f"Failed to create subscription record: {event['data_model']['object']['id']}")
    elif event['type'] == 'customer.subscription.updated':
        success = await update_subscription_record(event['data_model']['object'])
        print(f"Subscription updated: {event['data_model']['object']['id']} - Status: {event['data_model']['object']['status']}" if success else f"Failed to update subscription record: {event['data_model']['object']['id']}")
    elif event['type'] == 'customer.subscription.deleted':
        success = await cancel_subscription_record(event['data_model']['object'])
        print(f"Subscription cancelled: {event['data_model']['object']['id']}" if success else f"Failed to cancel subscription record: {event['data_model']['object']['id']}")
    elif event['type'] == 'invoice.payment_succeeded':
        success = await save_successful_invoice(event['data_model']['object'])
        print(f"Invoice saved: {event['data_model']['object']['id']}" if success else f"Failed to save invoice: {event['data_model']['object']['id']}")
    elif event['type'] == 'invoice.payment_failed':
        success = await save_failed_invoice(event['data_model']['object'])
        print(f"Failed payment recorded: {event['data_model']['object']['id']}" if success else f"Failed to record failed payment: {event['data_model']['object']['id']}")
    
    return {"status": "success"}


@router.get("/my-subscription")
async def get_my_subscription(
    current_user_email: str = Depends(get_current_user_email),
    user_service: UserService = Depends(get_user_service)
):
    """Get current user's active subscription"""
    user_doc = await user_service.get_user_by_email(current_user_email)
    if not user_doc:
        raise HTTPException(status_code=404, detail="User not found")
    
    subscription = await get_user_subscription(str(user_doc['_id']))
    return {"subscription": subscription}


@router.get("/my-invoices")
async def get_my_invoices(
    current_user_email: str = Depends(get_current_user_email),
    user_service: UserService = Depends(get_user_service)
):
    """Get current user's invoice history"""
    user_doc = await user_service.get_user_by_email(current_user_email)
    if not user_doc:
        raise HTTPException(status_code=404, detail="User not found")
    
    invoices = await get_user_invoices(str(user_doc['_id']))
    return {"invoices": invoices}