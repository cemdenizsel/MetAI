from utils.email_utils import send_welcome_email
from unittest import TestCase


class TestEmailService(TestCase):
    def test_send_welcome_email(self):
        send_welcome_email("dogukangundogan5@gmail.com","Dogukan")