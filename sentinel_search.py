
import urllib.request
import urllib.parse
import json

# Study area: Shivamogga, Karnataka
LATITUDE = 13.9299
LONGITUDE = 75.5681

# Search Sentinel-2 L2A products
params = {
    "$top": "10",
    "$filter": (
        "Collection/Name eq 'SENTINEL-2' "
        "and contains(Name,'MSIL2A') "
        "and OData.CSC.Intersects("
        "area=geography'SRID=4326;POINT("
        f"{LONGITUDE} {LATITUDE}"
        ")')"
    ),
    "$orderby": "ContentDate/Start desc"
}

base_url = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
url = base_url + "?" + urllib.parse.urlencode(params)

print("Searching Sentinel-2 imagery...")
print("Study Area: Shivamogga, Karnataka")

try:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Satellite-Change-Detection-Project"}
    )

    with urllib.request.urlopen(request, timeout=60) as response:
        data = json.loads(response.read().decode())

    products = data.get("value", [])

    print("\nHTTP Status: 200")
    print("Products found:", len(products))

    for product in products:
        print("\nProduct:", product["Name"])
        print("ID:", product["Id"])
        print("Online:", product.get("Online"))

except Exception as error:
    print("\nSearch failed:", error)
