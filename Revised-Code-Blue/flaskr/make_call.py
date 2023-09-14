import os 
from twilio.rest import Client 

account_sid = 'ACcdb2803bbe0ef91809cb4f49a534826c'
auth_token = '784190f30c6248db427397d4fbef8853'

client = Client(account_sid, auth_token)

call = client.calls.create(

   to = "+18722233043",
   from_= "+18445390311",
   url="http://demo.twilio.com/docs/voice.xml"
)

print(call.sid)