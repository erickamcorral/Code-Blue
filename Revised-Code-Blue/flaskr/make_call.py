import os 
from twilio.rest import Client 

account_sid ='ACc305b5682b0417233eec381e16146126'
auth_token = '81791c95af08e6f4d9b45815a27f4290'

client = Client(account_sid, auth_token)

call = client.calls.create(

   to = "+18722233043",
   from_= "+18449744305",
   url="http://demo.twilio.com/docs/voice.xml"
)

print(call.sid)