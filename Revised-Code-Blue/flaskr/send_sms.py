# Download the helper library from https://www.twilio.com/docs/python/install
import os
from twilio.rest import Client


# Find your Account SID and Auth Token at twilio.com/console
# and set the environment variables. See http://twil.io/secure
account_sid = 'ACcdb2803bbe0ef91809cb4f49a534826c'
auth_token = '784190f30c6248db427397d4fbef8853'
client = Client(account_sid, auth_token)

message = client.messages \
    .create(
         body='This is the ship that made the Kessel Run in fourteen parsecs?',
         from_='+18445390311',
         to='+18722233043'
     )

print(message.sid)