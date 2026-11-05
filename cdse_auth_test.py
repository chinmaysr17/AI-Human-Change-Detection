
import requests
from getpass import getpass

TOKEN_URL = (
    "https://identity.dataspace.copernicus.eu/"
    "auth/realms/CDSE/protocol/openid-connect/token"
)

def main():
    username = input("Copernicus username/email: ")
    password = getpass("Copernicus password (hidden): ")

    payload = {
        "client_id": "cdse-public",
        "username": username,
        "password": password,
        "grant_type": "password"
    }

    try:
        response = requests.post(
            TOKEN_URL,
            data=payload,
            timeout=30
        )

        if response.status_code == 200:
            token_data = response.json()

            print("\nAuthentication successful!")
            print("Access token received.")
            print("Token type:", token_data.get("token_type"))
            print("Expires in:", token_data.get("expires_in"), "seconds")

            # Do not print or save the access token.
            del token_data
        else:
            print("\nAuthentication failed.")
            print("HTTP Status:", response.status_code)
            print("Check your account credentials and try again.")

    except requests.RequestException as error:
        print("\nConnection error:", error)

    finally:
        password = None

if __name__ == "__main__":
    main()
