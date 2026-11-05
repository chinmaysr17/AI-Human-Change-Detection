
import requests
from getpass import getpass

TOKEN_URL = (
    "https://identity.dataspace.copernicus.eu/"
    "auth/realms/CDSE/protocol/openid-connect/token"
)

PRODUCT_ID = "858fe95e-2d5b-4230-9d34-3ebaf7208221"

DOWNLOAD_URL = (
    "https://download.dataspace.copernicus.eu/"
    f"odata/v1/Products({PRODUCT_ID})/$value"
)

username = input("Copernicus username/email: ")
password = getpass("Copernicus password (hidden): ")

try:
    token_response = requests.post(
        TOKEN_URL,
        data={
            "client_id": "cdse-public",
            "username": username,
            "password": password,
            "grant_type": "password"
        },
        timeout=30
    )
    token_response.raise_for_status()

    access_token = token_response.json()["access_token"]

    print("\nAuthentication successful.")
    print("Checking product download endpoint...")

    with requests.get(
        DOWNLOAD_URL,
        headers={"Authorization": f"Bearer {access_token}"},
        stream=True,
        timeout=60
    ) as response:

        print("HTTP Status:", response.status_code)
        print("Content Type:", response.headers.get("Content-Type"))
        print("Content Length:", response.headers.get("Content-Length"))
        print("Final URL:", response.url.split("?")[0])

        if response.status_code == 200:
            print("Download endpoint accessible.")
        elif response.status_code == 206:
            print("Partial content accepted.")
        else:
            print("Download endpoint needs further investigation.")

except requests.RequestException as error:
    print("Request failed:", error)

finally:
    password = None
    access_token = None
