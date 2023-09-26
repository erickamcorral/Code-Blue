# Download the helper library from https://www.twilio.com/docs/python/install
import os
from twilio.rest import Client


# Find your Account SID and Auth Token at twilio.com/console
# and set the environment variables. See http://twil.io/secure
account_sid ='ACc305b5682b0417233eec381e16146126'
auth_token = '81791c95af08e6f4d9b45815a27f4290'
client = Client(account_sid, auth_token)

message = client.messages \
    .create(
         body='CODE BLUE HAS DETECTED POTENTIALLY DANGEROUS STROKE SYMPTOMS FOR <User>. PLEASE SEEK MEDICAL ASSISTANCE IMMEDIATELY.',
         from_='+18449744305',
         to='+18722233043'
     )

print(message.sid)