
// =====================================================
// MAP CONFIGURATION
// =====================================================

const DEFAULT_MAP_LATITUDE = 13.9299;
const DEFAULT_MAP_LONGITUDE = 75.5681;
const DEFAULT_MAP_ZOOM = 12;
const DEFAULT_ANALYSIS_RADIUS = 5;

let locationMap = null;
let selectedMarker = null;
let selectedCircle = null;
let selectedLocation = null;


// =====================================================
// STATUS MESSAGE
// =====================================================

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


// =====================================================
// LOCATION DETAILS
// =====================================================

function updateLocationDetails(latitude, longitude) {
    document.getElementById("selectedLatitude").textContent =
        latitude.toFixed(6);

    document.getElementById("selectedLongitude").textContent =
        longitude.toFixed(6);

    document.getElementById("selectedRadius").textContent =
        `${DEFAULT_ANALYSIS_RADIUS} km`;
}


// =====================================================
// SELECT LOCATION
// =====================================================

function selectAnalysisLocation(
    latitude,
    longitude,
    label = "Selected location"
) {
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
        "Location selected. Choose both dates and click Analyze."
    );
}


// =====================================================
// INITIALIZE MAP
// =====================================================

function initializeMap() {
    const mapElement = document.getElementById("locationMap");

    if (!mapElement) {
        console.error("Map container #locationMap was not found.");
        return;
    }

    if (typeof L === "undefined") {
        showSelectionStatus(
            "Map library did not load. Check your internet connection.",
            true
        );
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

    L.tileLayer(
        "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
        {
            attribution:
                '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
            maxZoom: 19
        }
    ).addTo(locationMap);

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


// =====================================================
// SEARCH LOCATION
// =====================================================

async function searchForLocation() {
    const input = document.getElementById("locationSearchInput");
    const resultsContainer =
        document.getElementById("locationSearchResults");

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
        const url = new URL(
            "https://nominatim.openstreetmap.org/search"
        );

        url.searchParams.set("q", query);
        url.searchParams.set("format", "jsonv2");
        url.searchParams.set("limit", "5");

        const response = await fetch(url.toString());

        if (!response.ok) {
            throw new Error("Location search service returned an error.");
        }

        const results = await response.json();
        resultsContainer.replaceChildren();

        if (!results.length) {
            const message = document.createElement("p");
            message.textContent = "No matching locations found.";
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
            "Could not search for this location. Please try again.";

        resultsContainer.appendChild(message);

        showSelectionStatus(
            "Location search failed. Please try again.",
            true
        );
    }
}


// =====================================================
// CREATE ANALYSIS RESULTS SECTION
// =====================================================

function getAnalysisResultsContainer() {
    let container = document.getElementById("sentinelAnalysisResults");

    if (container) {
        return container;
    }

    container = document.createElement("section");
    container.id = "sentinelAnalysisResults";

    container.style.marginTop = "25px";
    container.style.padding = "20px";
    container.style.borderRadius = "12px";
    container.style.background = "#ffffff";
    container.style.border = "1px solid #e5e7eb";
    container.style.display = "none";

    const mapElement = document.getElementById("locationMap");

    if (mapElement && mapElement.parentElement) {
        mapElement.parentElement.insertAdjacentElement(
            "afterend",
            container
        );
    } else {
        document.body.appendChild(container);
    }

    return container;
}


// =====================================================
// DISPLAY ANALYSIS RESULTS
// =====================================================

function displayAnalysisResults(result) {
    const container = getAnalysisResultsContainer();

    container.replaceChildren();
    container.style.display = "block";

    const heading = document.createElement("h2");
    heading.textContent = "Satellite Change Detection Results";
    container.appendChild(heading);

    const warning = document.createElement("p");
    warning.textContent = result.validation_status ||
        "Experimental prediction: model performance on Sentinel-2 has not been validated.";

    warning.style.padding = "12px";
    warning.style.background = "#fff7ed";
    warning.style.color = "#9a3412";
    warning.style.borderRadius = "8px";

    container.appendChild(warning);

    const statistics = document.createElement("div");
    statistics.style.display = "grid";
    statistics.style.gridTemplateColumns =
        "repeat(auto-fit, minmax(150px, 1fr))";
    statistics.style.gap = "12px";
    statistics.style.margin = "20px 0";

    const statItems = [
        ["Changed pixels", result.predicted_change_pixels],
        ["Total pixels", result.total_pixels],
        ["Predicted change", `${Number(result.change_percentage).toFixed(2)}%`],
        ["Threshold", result.threshold],
        ["Minimum probability", Number(result.prediction_min).toFixed(4)],
        ["Maximum probability", Number(result.prediction_max).toFixed(4)]
    ];

    statItems.forEach(function ([label, value]) {
        const card = document.createElement("div");

        card.style.padding = "14px";
        card.style.border = "1px solid #e5e7eb";
        card.style.borderRadius = "8px";

        const title = document.createElement("div");
        title.textContent = label;
        title.style.fontSize = "13px";
        title.style.color = "#64748b";

        const number = document.createElement("strong");
        number.textContent = value;
        number.style.display = "block";
        number.style.fontSize = "20px";
        number.style.marginTop = "6px";

        card.append(title, number);
        statistics.appendChild(card);
    });

    container.appendChild(statistics);

    const imageGrid = document.createElement("div");

    imageGrid.style.display = "grid";
    imageGrid.style.gridTemplateColumns =
        "repeat(auto-fit, minmax(240px, 1fr))";
    imageGrid.style.gap = "18px";

    const imageItems = [
        ["Before Image", result.before_image],
        ["After Image", result.after_image],
        ["Predicted Change Map", result.change_map],
        ["Change Overlay", result.change_overlay]
    ];

    imageItems.forEach(function ([label, source]) {
        const card = document.createElement("div");

        const title = document.createElement("h3");
        title.textContent = label;

        const image = document.createElement("img");
        image.src = source;
        image.alt = label;
        image.style.width = "100%";
        image.style.height = "auto";
        image.style.borderRadius = "8px";
        image.style.border = "1px solid #e5e7eb";

        card.append(title, image);
        imageGrid.appendChild(card);
    });

    container.appendChild(imageGrid);

    const record = document.createElement("p");
    record.textContent = `Analysis record ID: ${result.record_id}`;
    record.style.marginTop = "20px";
    record.style.color = "#64748b";

    container.appendChild(record);

    container.scrollIntoView({
        behavior: "smooth",
        block: "start"
    });
}


// =====================================================
// RUN LOCATION-BASED ANALYSIS
// =====================================================

async function validateAnalysisSelection() {

    // Require authentication before starting satellite analysis.
    if (!window.currentUser) {
        showSelectionStatus(
            "Please log in before starting satellite image analysis.",
            true
        );

        const authSection = document.getElementById("authSection");

        if (authSection) {
            authSection.scrollIntoView({
                behavior: "smooth",
                block: "center"
            });
        }

        return;
    }

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
        radius_km: DEFAULT_ANALYSIS_RADIUS,
        before_date: beforeDate,
        after_date: afterDate
    };

    const button = document.getElementById("validateAnalysisButton");
    const originalButtonText = button.textContent;

    button.disabled = true;
    button.textContent = "Analyzing...";

    try {
        if (typeof analyzeLocation !== "function") {
            throw new Error(
                "analyzeLocation() is unavailable. Check frontend/js/api.js."
            );
        }

        showSelectionStatus(
            "Retrieving Sentinel-2 images and running experimental change detection. Please wait."
        );

        const locationStatus = document.getElementById("locationStatus");

        if (locationStatus) {
            locationStatus.textContent = "Satellite analysis in progress...";
        }

        const resultsContainer = getAnalysisResultsContainer();
        resultsContainer.style.display = "block";
        resultsContainer.textContent =
            "Downloading satellite imagery and processing the selected dates. This may take several minutes.";

        const result = await analyzeLocation(requestBody);

        window.getCurrentAnalysisSelection = function () {
            return { ...requestBody };
        };

        displayAnalysisResults(result);

        showSelectionStatus(
            "Satellite analysis completed. Review the experimental results below."
        );

        if (locationStatus) {
            locationStatus.textContent = "Satellite analysis completed.";
        }

        console.log("Sentinel-2 analysis result:", result);

    } catch (error) {
        console.error("Location analysis failed:", error);

        showSelectionStatus(
            `Analysis failed: ${error.message}`,
            true
        );

        const resultsContainer = getAnalysisResultsContainer();
        resultsContainer.style.display = "none";

    } finally {
        button.disabled = false;
        button.textContent = originalButtonText;
    }
}


// =====================================================
// INITIALIZE DATE AND BUTTON EVENTS
// =====================================================

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


// =====================================================
// PAGE INITIALIZATION
// =====================================================

document.addEventListener("DOMContentLoaded", function () {
    initializeMap();
    initializeDateSelection();
});
