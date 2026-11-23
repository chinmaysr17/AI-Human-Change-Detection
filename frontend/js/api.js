
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

            // Required for HttpOnly authentication cookies
            credentials: "include",

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
// AUTHENTICATION
// Backend: /api/auth
// =====================================================

// REGISTER USER

async function registerUser(userData) {
    if (!userData || typeof userData !== "object") {
        throw new Error("Invalid registration details.");
    }

    if (!userData.full_name || !userData.email || !userData.password) {
        throw new Error("Please provide your name, email and password.");
    }

    return await apiRequest("/api/auth/register", {
        method: "POST",
        body: JSON.stringify({
            full_name: userData.full_name.trim(),
            email: userData.email.trim(),
            password: userData.password
        })
    });
}


// LOGIN USER

async function loginUser(credentials) {
    if (!credentials || typeof credentials !== "object") {
        throw new Error("Invalid login details.");
    }

    if (!credentials.email || !credentials.password) {
        throw new Error("Please enter your email and password.");
    }

    return await apiRequest("/api/auth/login", {
        method: "POST",
        body: JSON.stringify({
            email: credentials.email.trim(),
            password: credentials.password
        })
    });
}


// LOGOUT USER

async function logoutUser() {
    return await apiRequest("/api/auth/logout", {
        method: "POST"
    });
}


// GET CURRENT LOGGED-IN USER

async function getCurrentUser() {
    return await apiRequest("/api/auth/me");
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
// LOCATION-BASED SENTINEL-2 ANALYSIS
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
// GET USER-SPECIFIC ANALYSIS HISTORY
// =====================================================

async function getAnalysisHistory() {
    return await apiRequest("/api/analysis-history");
}
