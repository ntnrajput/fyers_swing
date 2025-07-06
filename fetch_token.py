from fyers_apiv3 import fyersModel
import webbrowser

# User-specific details
client_id = "VE3CCLJZWA-100"    # Example: 'UABC123XYZ-100'
secret_key = "QEGA69PVUL"
redirect_uri = "https://www.google.com"  # Example: 'https://www.yourdomain.com/callback'
response_type = "code"  
grant_type = "authorization_code"  

auth_code= "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhcHBfaWQiOiJWRTNDQ0xKWldBIiwidXVpZCI6IjUwN2ZlMjVkMjRhOTQxNDJiYTZmYjM3YjI5NDQ0YTgwIiwiaXBBZGRyIjoiIiwibm9uY2UiOiIiLCJzY29wZSI6IiIsImRpc3BsYXlfbmFtZSI6IlhUMDI2MjQiLCJvbXMiOiJLMSIsImhzbV9rZXkiOiI5YjViNjVmY2VmMzliNjJjZDlkZjBjZmU4YzhjYmRlMDk3ZDQxYmRkMGRlMmFiNWZlZjgwYWZjYyIsImlzRGRwaUVuYWJsZWQiOiJOIiwiaXNNdGZFbmFibGVkIjoiTiIsImF1ZCI6IltcImQ6MVwiLFwiZDoyXCIsXCJ4OjBcIixcIng6MVwiLFwieDoyXCJdIiwiZXhwIjoxNzUxODI5NDczLCJpYXQiOjE3NTE3OTk0NzMsImlzcyI6ImFwaS5sb2dpbi5meWVycy5pbiIsIm5iZiI6MTc1MTc5OTQ3Mywic3ViIjoiYXV0aF9jb2RlIn0.jaruS7A6sb-HbID1Kh-UZ5SLUakDZb8kQf1butKHivs"

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