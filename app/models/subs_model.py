from pydantic import BaseModel
from typing import Optional

class CheckoutRequest(BaseModel):
    lookup_key: str
    customer_email: Optional[str] = None
