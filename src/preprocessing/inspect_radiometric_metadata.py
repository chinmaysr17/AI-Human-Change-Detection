
import os
import xml.etree.ElementTree as ET

RAW_DIR = "data/sentinel2/raw"

DATES = ["20260927", "20261027"]

TARGET_TAGS = {
    "PRODUCT_START_TIME",
    "PRODUCT_STOP_TIME",
    "PROCESSING_BASELINE",
    "BOA_QUANTIFICATION_VALUE",
}


def inspect_product(date):
    print("\n" + "=" * 65)
    print(f"SENTINEL-2 METADATA — {date}")
    print("=" * 65)

    safe_folder = None

    for name in os.listdir(RAW_DIR):
        if date in name and name.endswith(".SAFE"):
            safe_folder = os.path.join(RAW_DIR, name)
            break

    if safe_folder is None:
        print("ERROR: SAFE folder not found.")
        return

    xml_path = os.path.join(safe_folder, "MTD_MSIL2A.xml")

    if not os.path.exists(xml_path):
        print("ERROR: Metadata XML not found:", xml_path)
        return

    root = ET.parse(xml_path).getroot()

    print("\nPRODUCT METADATA:")

    for element in root.iter():
        tag = element.tag.split("}")[-1]

        if tag in TARGET_TAGS:
            print(f"{tag}: {element.text}")

    print("\nBOA ADDITIVE OFFSETS:")

    offset_count = 0

    for element in root.iter():
        tag = element.tag.split("}")[-1]

        if tag == "BOA_ADD_OFFSET":
            band_id = element.attrib.get("band_id", "Unknown")
            print(f"Band ID {band_id}: {element.text}")
            offset_count += 1

    if offset_count == 0:
        print("No BOA_ADD_OFFSET entries found.")

    print(f"\nTotal offset entries: {offset_count}")


if __name__ == "__main__":
    for date in DATES:
        inspect_product(date)
