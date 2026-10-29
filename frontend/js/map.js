
const DEFAULT_MAP_LATITUDE = 13.9299;
const DEFAULT_MAP_LONGITUDE = 75.5681;
const DEFAULT_MAP_ZOOM = 12;
const DEFAULT_ANALYSIS_RADIUS = 5;

let locationMap = null;
let selectedMarker = null;
let selectedCircle = null;
let selectedLocation = null;

function showSelectionStatus(message, isError = false) {
    const status = document.getElementById("dateSelectionStatus");

    if (!status) return;

    status.textContent = message;
    status.style.display = "block";
    status.style.visibility = "visible";
    status.style.opacity = "1";
    status.style.padding = "12px";
    status.style.marginTop = "12px";
    status.style.borderRadius = "8px";
    status.style.backgroundColor = isError ? "#fee2e2" : "#dcfce7";
    status.style.color = isError ? "#991b1b" : "#166534";
    status.style.border = isError
        ? "1px solid #fca5a5"
        : "1px solid #86efac";
    status.className = isError ? "status error" : "status success";
}

function updateLocationDetails(latitude, longitude) {
    document.getElementById("selectedLatitude").textContent =
        latitude.toFixed(6);

    document.getElementById("selectedLongitude").textContent =
        longitude.toFixed(6);

    document.getElementById("selectedRadius").textContent =
        `${DEFAULT_ANALYSIS_RADIUS} km`;
}

function selectAnalysisLocation(latitude, longitude, label = "Selected location") {
    selectedLocation = {
        latitude: Number(latitude),
        longitude: Number(longitude),
        label
    };

    if (locationMap) {
        locationMap.setView(
            [selectedLocation.latitude, selectedLocation.longitude],
            14
        );

        if (selectedMarker) {
            locationMap.removeLayer(selectedMarker);
        }

        if (selectedCircle) {
            locationMap.removeLayer(selectedCircle);
        }

        selectedMarker = L.marker([
            selectedLocation.latitude,
            selectedLocation.longitude
        ]).addTo(locationMap);

        selectedMarker.bindPopup(label).openPopup();

        selectedCircle = L.circle(
            [selectedLocation.latitude, selectedLocation.longitude],
            {
                radius: DEFAULT_ANALYSIS_RADIUS * 1000,
                color: "#2563eb",
                fillColor: "#3b82f6",
                fillOpacity: 0.12,
                weight: 2
            }
        ).addTo(locationMap);
    }

    updateLocationDetails(
        selectedLocation.latitude,
        selectedLocation.longitude
    );

    const locationStatus = document.getElementById("locationStatus");

    if (locationStatus) {
        locationStatus.textContent = `Selected location: ${label}`;
    }

    window.getCurrentAnalysisSelection = null;

    showSelectionStatus(
        "Location selected. Choose your dates and click Validate Selection."
    );
}

function initializeMap() {
    const mapElement = document.getElementById("locationMap");

    if (!mapElement) {
        console.error("Map container #locationMap was not found.");
        return;
    }

    if (typeof L === "undefined") {
        showSelectionStatus(
            "The map library did not load. Check your internet connection and refresh.",
            true
        );
        console.error("Leaflet is not loaded.");
        return;
    }

    if (locationMap) {
        locationMap.invalidateSize();
        return;
    }

    locationMap = L.map(mapElement).setView(
        [DEFAULT_MAP_LATITUDE, DEFAULT_MAP_LONGITUDE],
        DEFAULT_MAP_ZOOM
    );

    L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
        attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
        maxZoom: 19
    }).addTo(locationMap);

    locationMap.on("click", function (event) {
        selectAnalysisLocation(
            event.latlng.lat,
            event.latlng.lng,
            "Selected map point"
        );
    });

    selectAnalysisLocation(
        DEFAULT_MAP_LATITUDE,
        DEFAULT_MAP_LONGITUDE,
        "Default location — select your intended location"
    );

    locationMap.invalidateSize();
    console.log("Location map initialized successfully.");
}

async function searchForLocation() {
    const input = document.getElementById("locationSearchInput");
    const resultsContainer = document.getElementById("locationSearchResults");
    const query = input.value.trim();

    if (!query) {
        showSelectionStatus("Please enter a location to search.", true);
        return;
    }

    resultsContainer.replaceChildren();

    const loadingMessage = document.createElement("p");
    loadingMessage.textContent = "Searching for location...";
    resultsContainer.appendChild(loadingMessage);

    try {
        const url = new URL("https://nominatim.openstreetmap.org/search");
        url.searchParams.set("q", query);
        url.searchParams.set("format", "jsonv2");
        url.searchParams.set("limit", "5");

        const response = await fetch(url.toString());

        if (!response.ok) {
            throw new Error("The location search service returned an error.");
        }

        const results = await response.json();
        resultsContainer.replaceChildren();

        if (!results.length) {
            const message = document.createElement("p");
            message.textContent = "No matching locations found. Try another search.";
            resultsContainer.appendChild(message);
            return;
        }

        results.forEach(function (place) {
            const button = document.createElement("button");
            button.type = "button";
            button.textContent = place.display_name;
            button.style.display = "block";
            button.style.width = "100%";
            button.style.textAlign = "left";
            button.style.padding = "10px";
            button.style.marginTop = "5px";
            button.style.cursor = "pointer";

            button.addEventListener("click", function () {
                selectAnalysisLocation(
                    Number(place.lat),
                    Number(place.lon),
                    place.display_name
                );

                input.value = place.display_name;
                resultsContainer.replaceChildren();
            });

            resultsContainer.appendChild(button);
        });
    } catch (error) {
        console.error("Location search failed:", error);
        resultsContainer.replaceChildren();

        const message = document.createElement("p");
        message.textContent =
            "Could not search for this location. Check your internet connection and try again.";
        resultsContainer.appendChild(message);

        showSelectionStatus(
            "Location search failed. Please try again.",
            true
        );
    }
}

async function validateAnalysisSelection() {
    const beforeDate = document.getElementById("beforeDate").value;
    const afterDate = document.getElementById("afterDate").value;

    if (!selectedLocation) {
        showSelectionStatus(
            "Please select a location on the map or search for one first.",
            true
        );
        return;
    }

    if (!beforeDate || !afterDate) {
        showSelectionStatus(
            "Please select both the before date and the after date.",
            true
        );
        return;
    }

    if (beforeDate >= afterDate) {
        showSelectionStatus(
            "The before date must be earlier than the after date.",
            true
        );
        return;
    }

    const requestBody = {
        latitude: selectedLocation.latitude,
        longitude: selectedLocation.longitude,
        location: selectedLocation.label,
        radius_km: DEFAULT_ANALYSIS_RADIUS,
        before_date: beforeDate,
        after_date: afterDate
    };

    const button = document.getElementById("validateAnalysisButton");
    button.disabled = true;

    try {
        if (typeof saveAnalysisSelection !== "function") {
            throw new Error(
                "saveAnalysisSelection() is unavailable. Check js/api.js."
            );
        }

        await saveAnalysisSelection(requestBody);

        window.getCurrentAnalysisSelection = function () {
            return { ...requestBody };
        };

        showSelectionStatus(
            `Selection saved successfully! Location: ${selectedLocation.label}. ` +
            `Before: ${beforeDate}; After: ${afterDate}. ` +
            "This saves the location and dates; it does not retrieve satellite imagery."
        );

        const locationStatus = document.getElementById("locationStatus");
        if (locationStatus) {
            locationStatus.textContent = "Location and dates validated successfully.";
        }

        console.log("Analysis selection saved:", requestBody);
    } catch (error) {
        console.error("Could not save analysis selection:", error);

        showSelectionStatus(
            `Could not save the selection: ${error.message}`,
            true
        );
    } finally {
        button.disabled = false;
    }
}

function initializeDateSelection() {
    const beforeDateInput = document.getElementById("beforeDate");
    const afterDateInput = document.getElementById("afterDate");

    const today = new Date();
    const todayString = [
        today.getFullYear(),
        String(today.getMonth() + 1).padStart(2, "0"),
        String(today.getDate()).padStart(2, "0")
    ].join("-");

    beforeDateInput.max = todayString;
    afterDateInput.max = todayString;

    document
        .getElementById("locationSearchButton")
        .addEventListener("click", searchForLocation);

    document
        .getElementById("locationSearchInput")
        .addEventListener("keydown", function (event) {
            if (event.key === "Enter") {
                event.preventDefault();
                searchForLocation();
            }
        });

    document
        .getElementById("validateAnalysisButton")
        .addEventListener("click", validateAnalysisSelection);
}

document.addEventListener("DOMContentLoaded", function () {
    initializeMap();
    initializeDateSelection();
});
