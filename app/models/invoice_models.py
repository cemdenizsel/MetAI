from typing import Optional

from bson import ObjectId
from pydantic import BaseModel, Field

class Invoice(BaseModel):
    id: ObjectId = Field(description= "Id of the invoice")
    user_id: ObjectId = Field(description= "Id of the user")
    is_paid: bool = Field(description= "Is the invoice paid")
    payment_token: str = Field(description= "Payment token")
    provider: str = Field(description= "Provider (Google, Apple or Stripe)")


