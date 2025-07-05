from fyers_apiv3 import fyersModel
import webbrowser

# User-specific details
client_id = "VE3CCLJZWA-100"    # Example: 'UABC123XYZ-100'
secret_key = "QEGA69PVUL"
redirect_uri = "https://www.google.com"  # Example: 'https://www.yourdomain.com/callback'
response_type = "code"  
grant_type = "authorization_code"  

auth_code= "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhcHBfaWQiOiJWRTNDQ0xKWldBIiwidXVpZCI6IjhjN2Y2NmU5MDNlNDQ5ZDhiMjdjODU0N2I0MmI2NmIxIiwiaXBBZGRyIjoiIiwibm9uY2UiOiIiLCJzY29wZSI6IiIsImRpc3BsYXlfbmFtZSI6IlhUMDI2MjQiLCJvbXMiOiJLMSIsImhzbV9rZXkiOiI5YjViNjVmY2VmMzliNjJjZDlkZjBjZmU4YzhjYmRlMDk3ZDQxYmRkMGRlMmFiNWZlZjgwYWZjYyIsImlzRGRwaUVuYWJsZWQiOiJOIiwiaXNNdGZFbmFibGVkIjoiTiIsImF1ZCI6IltcImQ6MVwiLFwiZDoyXCIsXCJ4OjBcIixcIng6MVwiLFwieDoyXCJdIiwiZXhwIjoxNzUxNzU2NzYwLCJpYXQiOjE3NTE3MjY3NjAsImlzcyI6ImFwaS5sb2dpbi5meWVycy5pbiIsIm5iZiI6MTc1MTcyNjc2MCwic3ViIjoiYXV0aF9jb2RlIn0.k9vTpyeUDcM-HZ36yxlCj0nbZLYQysyqlgjw7Fy86rY"
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