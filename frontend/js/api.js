
// =====================================================
// API CONFIGURATION
// =====================================================

const API_BASE_URL = "http://127.0.0.1:8000";


// =====================================================
// COMMON API REQUEST FUNCTION
// =====================================================

async function apiRequest(endpoint, options = {}) {
    const url = `${API_BASE_URL}${endpoint}`;

    let response;

    try {
        response = await fetch(url, {
            ...options,
            headers: {
                "Content-Type": "application/json",
                ...(options.headers || {})
            }
        });
    } catch (error) {
        throw new Error(
            "Cannot connect to the backend. Make sure FastAPI is running at http://127.0.0.1:8000."
        );
    }

    let data;

    try {
        data = await response.json();
    } catch (error) {
        throw new Error(
            `Invalid backend response (HTTP ${response.status}).`
        );
    }

    if (!response.ok) {
        const message =
            data.detail ||
            data.message ||
            `Request failed with status ${response.status}.`;

        throw new Error(
            typeof message === "string"
                ? message
                : JSON.stringify(message)
        );
    }

    return data;
}


// =====================================================
// GET MODEL INFORMATION
// =====================================================

async function getModelInformation() {
    return await apiRequest("/api/model");
}


// =====================================================
// ORIGINAL LEVIR-CD IMAGE ANALYSIS
// =====================================================

async function analyzeSatelliteImage(filename) {
    if (!filename || !filename.trim()) {
        throw new Error("Please enter a dataset image filename.");
    }

    const safeFilename = filename.trim();

    return await apiRequest(
        `/api/analyze/${encodeURIComponent(safeFilename)}`
    );
}


// =====================================================
// SAVE LOCATION AND DATE SELECTION
// =====================================================

async function saveAnalysisSelection(selection) {
    if (!selection || typeof selection !== "object") {
        throw new Error("Invalid analysis selection.");
    }

    return await apiRequest("/api/analysis-selection", {
        method: "POST",
        body: JSON.stringify(selection)
    });
}


// =====================================================
// NEW: LOCATION-BASED SENTINEL-2 ANALYSIS
// Backend: POST /api/analyze-location
// =====================================================

async function analyzeLocation(selection) {
    if (!selection || typeof selection !== "object") {
        throw new Error("Invalid location analysis request.");
    }

    const requiredFields = [
        "latitude",
        "longitude",
        "radius_km",
        "before_date",
        "after_date"
    ];

    for (const field of requiredFields) {
        if (selection[field] === undefined || selection[field] === null) {
            throw new Error(`Missing required field: ${field}`);
        }
    }

    return await apiRequest("/api/analyze-location", {
        method: "POST",
        body: JSON.stringify({
            latitude: Number(selection.latitude),
            longitude: Number(selection.longitude),
            radius_km: Number(selection.radius_km),
            before_date: selection.before_date,
            after_date: selection.after_date
        })
    });
}


// =====================================================
// GET ANALYSIS HISTORY
// =====================================================

async function getAnalysisHistory() {
    return await apiRequest("/api/analysis-history");
}
