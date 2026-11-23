
// =====================================================
// ANALYSIS HISTORY STATUS
// =====================================================

function showHistoryStatus(message, isError = false) {
    const status = document.getElementById("analysisHistoryStatus");

    if (!status) return;

    status.textContent = message;
    status.className = isError
        ? "history-status error"
        : "history-status";
}


// =====================================================
// FORMAT HISTORY VALUES
// =====================================================

function formatHistoryDate(value) {
    if (!value) return "--";

    const date = new Date(value);

    if (Number.isNaN(date.getTime())) {
        return String(value);
    }

    return date.toLocaleString();
}


// =====================================================
// DISPLAY ONE HISTORY RECORD
// =====================================================

function createHistoryCard(record, index) {
    const card = document.createElement("article");
    card.className = "history-card";

    const heading = document.createElement("h3");

    heading.className = "history-card-heading";

    heading.textContent =
        record.filename ||
        record.image_filename ||
        `Analysis ${record.id || index + 1}`;

    card.appendChild(heading);

    const details = [
        ["Record ID", record.id],
        ["Status", record.status],
        ["Location", record.location_name],
        ["Latitude", record.latitude],
        ["Longitude", record.longitude],
        ["Radius (km)", record.radius_km],
        ["Before Date", record.before_date],
        ["After Date", record.after_date],
        ["Change Percentage", record.change_percentage],
        ["Created At", record.created_at || record.timestamp]
    ];

    details.forEach(([label, value]) => {
        if (value === undefined || value === null || value === "") {
            return;
        }

        const row = document.createElement("p");
        row.className = "history-detail-row";
        const labelElement = document.createElement("strong");
        labelElement.className = "history-detail-label";
        const valueElement = document.createElement("span");
        valueElement.className = label === "Status"
            ? "history-detail-value history-status-badge"
            : "history-detail-value";

        labelElement.textContent = `${label}: `;

        valueElement.textContent =
            typeof value === "object"
                ? JSON.stringify(value)
                : String(value);

        row.appendChild(labelElement);
        row.appendChild(valueElement);
        card.appendChild(row);
    });

    const knownKeys = new Set([
        "id",
        "filename",
        "image_filename",
        "status",
        "location_name",
        "latitude",
        "longitude",
        "radius_km",
        "before_date",
        "after_date",
        "change_percentage",
        "created_at",
        "timestamp",
        "imagery_retrieved"
    ]);

    const otherEntries = Object.entries(record).filter(
        ([key, value]) =>
            !knownKeys.has(key) &&
            value !== undefined &&
            value !== null &&
            typeof value !== "object"
    );

    otherEntries.forEach(([key, value]) => {
        const row = document.createElement("p");
        const labelElement = document.createElement("strong");
        const valueElement = document.createElement("span");

        labelElement.textContent =
            `${key.replaceAll("_", " ")}: `;

        valueElement.textContent = String(value);

        row.appendChild(labelElement);
        row.appendChild(valueElement);
        card.appendChild(row);
    });

    if (record.imagery_retrieved !== undefined) {
        const row = document.createElement("p");
        const labelElement = document.createElement("strong");
        const valueElement = document.createElement("span");

        labelElement.textContent = "Imagery Retrieved: ";
        valueElement.textContent = record.imagery_retrieved
            ? "Yes"
            : "No";

        row.appendChild(labelElement);
        row.appendChild(valueElement);
        card.appendChild(row);
    }

    return card;
}


// =====================================================
// DISPLAY HISTORY RECORDS
// =====================================================

function displayAnalysisHistory(data) {
    const container = document.getElementById(
        "analysisHistoryContainer"
    );

    if (!container) return;

    container.replaceChildren();

    let records;

    if (Array.isArray(data)) {
        records = data;
    } else if (data && Array.isArray(data.history)) {
        records = data.history;
    } else if (data && Array.isArray(data.records)) {
        records = data.records;
    } else if (data && Array.isArray(data.items)) {
        records = data.items;
    } else {
        records = [];
    }

    if (records.length === 0) {
        const emptyMessage = document.createElement("div");

        emptyMessage.className = "history-empty";
        emptyMessage.textContent =
            "No analysis history records are available.";

        container.appendChild(emptyMessage);
        return;
    }

    records.forEach((record, index) => {
        if (!record || typeof record !== "object") return;

        container.appendChild(
            createHistoryCard(record, index)
        );
    });

    if (!container.children.length) {
        const emptyMessage = document.createElement("div");

        emptyMessage.className = "history-empty";
        emptyMessage.textContent =
            "No displayable history records were returned.";

        container.appendChild(emptyMessage);
    }
}


// =====================================================
// CLEAR HISTORY ON LOGOUT
// =====================================================

function clearAnalysisHistory() {
    const container = document.getElementById(
        "analysisHistoryContainer"
    );

    if (container) {
        container.replaceChildren();

        const message = document.createElement("div");
        message.className = "history-empty";
        message.textContent = "Please log in to view your analysis history.";

        container.appendChild(message);
    }

    showHistoryStatus("Login required to access analysis history.");
}


// =====================================================
// LOAD ANALYSIS HISTORY FROM THE BACKEND
// =====================================================

async function loadAnalysisHistory() {
    const button = document.getElementById("refreshHistoryButton");

    // Do not request protected history without authentication.
    if (!window.currentUser) {
        clearAnalysisHistory();
        return;
    }

    if (button) {
        button.disabled = true;
        button.textContent = "Loading...";
    }

    showHistoryStatus("Loading your analysis history...");

    try {
        const data = await getAnalysisHistory();

        // Check that the user is still logged in.
        if (!window.currentUser) {
            clearAnalysisHistory();
            return;
        }

        displayAnalysisHistory(data);

        showHistoryStatus(
            "Your analysis history loaded successfully."
        );
    } catch (error) {
        console.error("Could not load analysis history:", error);

        const container = document.getElementById(
            "analysisHistoryContainer"
        );

        if (container) {
            container.replaceChildren();

            const errorMessage = document.createElement("div");
            errorMessage.className = "history-empty";
            errorMessage.textContent =
                "History could not be loaded. Please try again.";
            
            container.appendChild(errorMessage);
        }

        showHistoryStatus(
            `Could not load history: ${error.message}`,
            true
        );
    } finally {
        if (button) {
            button.disabled = false;
            button.textContent = "Refresh History";
        }
    }
}


// =====================================================
// REGISTER HISTORY EVENTS
// =====================================================

document.addEventListener("DOMContentLoaded", () => {
    const refreshButton = document.getElementById(
        "refreshHistoryButton"
    );

    if (refreshButton) {
        refreshButton.addEventListener(
            "click",
            loadAnalysisHistory
        );
    }

    // History will be loaded by auth.js after successful login.
    clearAnalysisHistory();
});
