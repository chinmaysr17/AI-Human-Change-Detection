
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
    heading.textContent =
        record.filename ||
        record.image_filename ||
        `Analysis ${index + 1}`;

    card.appendChild(heading);

    const details = [
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
        const labelElement = document.createElement("strong");
        const valueElement = document.createElement("span");

        labelElement.textContent = `${label}: `;
        valueElement.textContent =
            typeof value === "object"
                ? JSON.stringify(value)
                : String(value);

        row.appendChild(labelElement);
        row.appendChild(valueElement);
        card.appendChild(row);
    });

    // Show other record fields if the backend uses a different schema.
    const knownKeys = new Set([
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
        "timestamp"
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

    // Support either a direct array or a common object wrapper.
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
// LOAD ANALYSIS HISTORY FROM THE BACKEND
// =====================================================

async function loadAnalysisHistory() {
    const button = document.getElementById("refreshHistoryButton");

    if (button) {
        button.disabled = true;
        button.textContent = "Loading...";
    }

    showHistoryStatus("Loading analysis history...");

    try {
        const data = await getAnalysisHistory();

        displayAnalysisHistory(data);

        showHistoryStatus("Analysis history loaded successfully.");
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
                "History could not be loaded. Check that the backend is running.";

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

    // Load history automatically when the page opens.
    loadAnalysisHistory();
});
