
// =====================================================
// ANALYSIS STATUS MESSAGES
// =====================================================

function showAnalysisStatus(message, isError = false) {
    const status = document.getElementById("status");

    status.textContent = message;
    status.className = isError
        ? "status loading error"
        : "status loading";
}


// =====================================================
// FORMAT NUMBERS SAFELY
// =====================================================

function formatNumber(value, decimals = 2) {
    const number = Number(value);

    return Number.isFinite(number)
        ? number.toLocaleString(undefined, {
            minimumFractionDigits: decimals,
            maximumFractionDigits: decimals
        })
        : "--";
}


// =====================================================
// DISPLAY ANALYSIS STATISTICS
// =====================================================

function displayAnalysisStatistics(data) {
    document.getElementById("changePercentage").textContent =
        `${formatNumber(data.change_percentage)}%`;

    document.getElementById("changedPixels").textContent =
        Number.isFinite(Number(data.predicted_change_pixels))
            ? Number(data.predicted_change_pixels).toLocaleString()
            : "--";

    document.getElementById("totalPixels").textContent =
        Number.isFinite(Number(data.total_pixels))
            ? Number(data.total_pixels).toLocaleString()
            : "--";

    const mean = Number(data.prediction_mean);

    document.getElementById("meanPrediction").textContent =
        Number.isFinite(mean)
            ? `${formatNumber(mean * 100)}%`
            : "--";

    document.getElementById("threshold").textContent =
        formatNumber(data.threshold);
}


// =====================================================
// DISPLAY AN IMAGE AND ITS PLACEHOLDER
// =====================================================

function displayAnalysisImage(imageId, placeholderId, imageSource) {
    const image = document.getElementById(imageId);
    const placeholder = document.getElementById(placeholderId);

    if (!image) return;

    if (typeof imageSource === "string" && imageSource.trim()) {
        image.onload = () => {
            image.hidden = false;

            if (placeholder) {
                placeholder.hidden = true;
            }
        };

        image.onerror = () => {
            image.hidden = true;

            if (placeholder) {
                placeholder.hidden = false;
                placeholder.textContent = "The image could not be displayed.";
            }
        };

        image.src = imageSource;
    } else {
        image.removeAttribute("src");
        image.hidden = true;

        if (placeholder) {
            placeholder.hidden = false;
            placeholder.textContent = "This image was not provided by the backend.";
        }
    }
}


// =====================================================
// DISPLAY DETECTED CHANGE REGIONS
// =====================================================

function displayChangeRegions(regions) {
    const container = document.getElementById("regionsContainer");
    const countElement = document.getElementById("regionCount");

    const validRegions = Array.isArray(regions) ? regions : [];

    countElement.textContent =
        `${validRegions.length} region${validRegions.length === 1 ? "" : "s"}`;

    container.replaceChildren();

    if (validRegions.length === 0) {
        const emptyMessage = document.createElement("div");

        emptyMessage.className = "no-regions";
        emptyMessage.textContent =
            "No detected change regions were returned by the model.";

        container.appendChild(emptyMessage);
        return;
    }

    const wrapper = document.createElement("div");
    wrapper.className = "regions-table-wrapper";

    const table = document.createElement("table");
    table.className = "regions-table";

    const headers = [
        "Region",
        "Area (Pixels)",
        "Area (%)",
        "Bounding Box (x, y, width, height)",
        "Bounding Box Area",
        "Aspect Ratio",
        "Fill Ratio",
        "Center (x, y)"
    ];

    const thead = document.createElement("thead");
    const headerRow = document.createElement("tr");

    headers.forEach((headerText) => {
        const th = document.createElement("th");
        th.textContent = headerText;
        headerRow.appendChild(th);
    });

    thead.appendChild(headerRow);
    table.appendChild(thead);

    const tbody = document.createElement("tbody");

    validRegions.forEach((region, index) => {
        const row = document.createElement("tr");
        const box = region.bounding_box || {};
        const center = region.center || {};

        const values = [
            index + 1,
            region.area_pixels,
            region.area_percentage,
            [
                box.x,
                box.y,
                box.width,
                box.height
            ].every(value => value !== undefined)
                ? `${box.x}, ${box.y}, ${box.width}, ${box.height}`
                : "--",
            region.bounding_box_area,
            region.aspect_ratio,
            region.fill_ratio,
            center.x !== undefined && center.y !== undefined
                ? `${center.x}, ${center.y}`
                : "--"
        ];

        values.forEach((value, columnIndex) => {
            const cell = document.createElement("td");

            if (columnIndex === 0) {
                const badge = document.createElement("span");
                badge.className = "region-number";
                badge.textContent = String(value);
                cell.appendChild(badge);
            } else if (
                value === undefined ||
                value === null ||
                (typeof value === "number" && !Number.isFinite(value))
            ) {
                cell.textContent = "--";
            } else if (
                [1, 4].includes(columnIndex) &&
                Number.isFinite(Number(value))
            ) {
                cell.textContent = Number(value).toLocaleString();
            } else if (
                [2, 5, 6].includes(columnIndex) &&
                Number.isFinite(Number(value))
            ) {
                cell.textContent = formatNumber(value, 3);
            } else {
                cell.textContent = String(value);
            }

            row.appendChild(cell);
        });

        tbody.appendChild(row);
    });

    table.appendChild(tbody);
    wrapper.appendChild(table);
    container.appendChild(wrapper);
}


// =====================================================
// DRAW DETECTED REGION BOXES ON THE AFTER IMAGE
// =====================================================

function drawRegionBoxes(regions, image) {
    const canvas = document.getElementById("regionCanvas");

    if (!canvas || !image) return;

    const context = canvas.getContext("2d");

    if (!context) return;

    const draw = () => {
        const width = image.naturalWidth;
        const height = image.naturalHeight;

        if (!width || !height) return;

        canvas.width = width;
        canvas.height = height;

        context.clearRect(0, 0, width, height);

        if (!Array.isArray(regions)) return;

        context.lineWidth = Math.max(2, width / 400);
        context.strokeStyle = "#ff3030";
        context.fillStyle = "#ff3030";
        context.font = `${Math.max(14, width / 45)}px Arial`;

        regions.forEach((region, index) => {
            const box = region.bounding_box;

            if (!box) return;

            const x = Number(box.x);
            const y = Number(box.y);
            const boxWidth = Number(box.width);
            const boxHeight = Number(box.height);

            if (
                ![x, y, boxWidth, boxHeight].every(Number.isFinite) ||
                boxWidth <= 0 ||
                boxHeight <= 0
            ) {
                return;
            }

            context.strokeRect(x, y, boxWidth, boxHeight);

            context.fillText(
                `R${index + 1}`,
                x,
                Math.max(14, y - 4)
            );
        });
    };

    if (image.complete && image.naturalWidth > 0) {
        draw();
    } else {
        image.addEventListener("load", draw, { once: true });
    }
}


// =====================================================
// RUN SATELLITE IMAGE ANALYSIS
// =====================================================

async function analyzeImage() {
    const filenameInput = document.getElementById("filename");
    const analyzeButton = document.getElementById("analyzeButton");

    const filename = filenameInput.value.trim();

    if (!filename) {
        showAnalysisStatus("Please enter a dataset image filename.", true);
        return;
    }

    analyzeButton.disabled = true;
    analyzeButton.textContent = "Analyzing...";

    showAnalysisStatus(
        "Running AI change detection. Please wait..."
    );

    try {
        // Call the shared API function from api.js.
        const data = await analyzeSatelliteImage(filename);

        // Display numerical results.
        displayAnalysisStatistics(data);

        // Display detected regions.
        const regions = Array.isArray(data.change_regions)
            ? data.change_regions
            : [];

        displayChangeRegions(regions);

        // Display satellite images and generated visualizations.
        displayAnalysisImage(
            "beforeImage",
            "beforeImagePlaceholder",
            data.before_image
        );

        displayAnalysisImage(
            "afterRegionImage",
            "afterImagePlaceholder",
            data.after_image
        );

        displayAnalysisImage(
            "changeMap",
            "changeMapPlaceholder",
            data.change_map
        );

        displayAnalysisImage(
            "changeOverlay",
            "changeOverlayPlaceholder",
            data.change_overlay
        );

        // Draw bounding boxes once the after image loads.
        const afterImage = document.getElementById("afterRegionImage");

        afterImage.onload = () => {
            afterImage.hidden = false;

            const placeholder = document.getElementById("afterImagePlaceholder");
            if (placeholder) placeholder.hidden = true;

            drawRegionBoxes(regions, afterImage);
        };

        // Reassign the image source so the box-drawing handler is attached.
        if (data.after_image) {
            afterImage.src = data.after_image;
        }

        showAnalysisStatus(
            `Analysis completed successfully for "${data.filename || filename}".`
        );

        console.log("Satellite image analysis result:", data);
    } catch (error) {
        console.error("Satellite image analysis failed:", error);

        showAnalysisStatus(
            `Analysis failed: ${error.message}`,
            true
        );
    } finally {
        analyzeButton.disabled = false;
        analyzeButton.textContent = "Analyze Image";
    }
}


// =====================================================
// REGISTER THE ANALYSIS BUTTON
// =====================================================

document.addEventListener("DOMContentLoaded", () => {
    document
        .getElementById("analyzeButton")
        .addEventListener("click", analyzeImage);
});
