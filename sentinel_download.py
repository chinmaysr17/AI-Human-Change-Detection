
import os
import requests
from getpass import getpass

TOKEN_URL = (
    "https://identity.dataspace.copernicus.eu/"
    "auth/realms/CDSE/protocol/openid-connect/token"
)

DOWNLOAD_BASE = "https://download.dataspace.copernicus.eu/odata/v1/Products"

PRODUCTS = {
    "2026-09-27": "95588b60-4a6d-4d0b-a17a-cd61ca5d2e5f",
    "2026-10-27": "858fe95e-2d5b-4230-9d34-3ebaf7208221"
}

OUTPUT_DIR = "data/sentinel2/raw"
CHUNK_SIZE = 1024 * 1024

def get_access_token():
    username = input("Copernicus username/email: ")
    password = getpass("Copernicus password (hidden): ")

    response = requests.post(
        TOKEN_URL,
        data={
            "client_id": "cdse-public",
            "username": username,
            "password": password,
            "grant_type": "password"
        },
        timeout=30
    )
    password = None
    response.raise_for_status()

    return response.json()["access_token"]


def download_product(token, date, product_id):
    filename = f"Sentinel2_{date}.zip"
    filepath = os.path.join(OUTPUT_DIR, filename)

    if os.path.exists(filepath):
        print(f"\nAlready exists: {filename}")
        return

    url = f"{DOWNLOAD_BASE}({product_id})/$value"

    print(f"\nDownloading: {date}")

    with requests.get(
        url,
        headers={"Authorization": f"Bearer {token}"},
        stream=True,
        timeout=(30, 180)
    ) as response:

        response.raise_for_status()

        total_size = int(response.headers.get("Content-Length", 0))
        downloaded = 0
        temporary_path = filepath + ".part"

        try:
            with open(temporary_path, "wb") as file:
                for chunk in response.iter_content(CHUNK_SIZE):
                    if chunk:
                        file.write(chunk)
                        downloaded += len(chunk)

                        if total_size:
                            percent = downloaded * 100 / total_size
                            print(
                                f"\rProgress: {percent:.1f}%",
                                end=""
                            )

            os.replace(temporary_path, filepath)
            print(f"\nSaved: {filepath}")

        except Exception:
            if os.path.exists(temporary_path):
                os.remove(temporary_path)
            raise


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    token = get_access_token()
    print("\nAuthentication successful.")

    try:
        for date, product_id in PRODUCTS.items():
            download_product(token, date, product_id)

        print("\nAll requested downloads completed.")

    except Exception as error:
        print("\nDownload error:", error)

    finally:
        token = None


if __name__ == "__main__":
    main()
