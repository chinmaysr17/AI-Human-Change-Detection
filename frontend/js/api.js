
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
            "Cannot connect to the backend. Make sure your FastAPI server is running at http://127.0.0.1:8000."
        );
    }

    let data;

    try {
        data = await response.json();
    } catch (error) {
        throw new Error(
            `The backend returned an invalid response (HTTP ${response.status}).`
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
// Backend endpoint: GET /api/model
// =====================================================

async function getModelInformation() {
    return await apiRequest("/api/model");
}


// =====================================================
// ANALYZE SATELLITE IMAGE
// Backend endpoint: GET /api/analyze/{filename}
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
// SAVE ANALYSIS LOCATION AND DATE SELECTION
// Backend endpoint: POST /api/analysis-selection
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
// GET ANALYSIS HISTORY
// Backend endpoint: GET /api/analysis-history
// =====================================================

async function getAnalysisHistory() {
    return await apiRequest("/api/analysis-history");
}
