from fyers_apiv3 import fyersModel
import webbrowser

# User-specific details
client_id = "VE3CCLJZWA-100"    # Example: 'UABC123XYZ-100'
secret_key = "QEGA69PVUL"
redirect_uri = "https://www.google.com"  # Example: 'https://www.yourdomain.com/callback'
response_type = "code"  
grant_type = "authorization_code"  

auth_code= "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhcHBfaWQiOiJWRTNDQ0xKWldBIiwidXVpZCI6IjM5MzZiNmYyOGFiODQyY2ZiNGIyYjMzYWZhYzA2NWZmIiwiaXBBZGRyIjoiIiwibm9uY2UiOiIiLCJzY29wZSI6IiIsImRpc3BsYXlfbmFtZSI6IlhUMDI2MjQiLCJvbXMiOiJLMSIsImhzbV9rZXkiOiI5YjViNjVmY2VmMzliNjJjZDlkZjBjZmU4YzhjYmRlMDk3ZDQxYmRkMGRlMmFiNWZlZjgwYWZjYyIsImlzRGRwaUVuYWJsZWQiOiJOIiwiaXNNdGZFbmFibGVkIjoiTiIsImF1ZCI6IltcImQ6MVwiLFwiZDoyXCIsXCJ4OjBcIixcIng6MVwiLFwieDoyXCJdIiwiZXhwIjoxNzUxMjAwMTcyLCJpYXQiOjE3NTExNzAxNzIsImlzcyI6ImFwaS5sb2dpbi5meWVycy5pbiIsIm5iZiI6MTc1MTE3MDE3Miwic3ViIjoiYXV0aF9jb2RlIn0.s7i2OgyAQPTlcGQNktzXWOZvYj6KtMXzxlMWPsblyOw"
session = fyersModel.SessionModel(
    client_id=client_id,
    secret_key=secret_key,
    redirect_uri=redirect_uri,
    response_type=response_type,
    grant_type=grant_type
)

session.set_token(auth_code)
response = session.generate_token()

print(response)