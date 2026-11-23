const MINING_API_BASE_URL = "http://127.0.0.1:8000";

document.addEventListener("DOMContentLoaded", () => {
    const splitSelect = document.getElementById("miningSplit");
    const tileSelect = document.getElementById("miningTile");
    const analyzeButton = document.getElementById("analyzeMiningButton");
    const statusElement = document.getElementById("miningAnalysisStatus");
    const resultsElement = document.getElementById("miningResults");

    if (!splitSelect || !tileSelect || !analyzeButton || !statusElement) {
        console.error("Mining section elements were not found.");
        return;
    }

    function setMiningStatus(message, type = "info") {
        statusElement.textContent = message;
        statusElement.className = `mining-status ${type}`;
    }

    function loadImage(source) {
        return new Promise((resolve, reject) => {
            const image = new Image();

            image.onload = () => resolve(image);
            image.onerror = () => reject(
                new Error("Could not load an image for the mining overlay.")
            );

            image.src = source;
        });
    }

    async function createMiningOverlay(rgbSource, maskSource) {
        const [rgbImage, maskImage] = await Promise.all([
            loadImage(rgbSource),
            loadImage(maskSource)
        ]);

        const width = rgbImage.naturalWidth;
        const height = rgbImage.naturalHeight;

        if (
            width !== maskImage.naturalWidth ||
            height !== maskImage.naturalHeight
        ) {
            throw new Error("RGB image and mining mask dimensions do not match.");
        }

        const canvas = document.createElement("canvas");
        canvas.width = width;
        canvas.height = height;

        const context = canvas.getContext("2d");

        context.drawImage(rgbImage, 0, 0, width, height);

        const rgbData = context.getImageData(0, 0, width, height);

        const maskCanvas = document.createElement("canvas");
        maskCanvas.width = width;
        maskCanvas.height = height;

        const maskContext = maskCanvas.getContext("2d");
        maskContext.drawImage(maskImage, 0, 0, width, height);

        const maskData = maskContext.getImageData(
            0, 0, width, height
        ).data;

        const overlayData = rgbData.data;
        const alpha = 0.50;

        for (let i = 0; i < overlayData.length; i += 4) {
            const isMining = maskData[i] > 127;

            if (isMining) {
                // Blend predicted mining pixels with red.
                overlayData[i] =
                    255 * alpha + overlayData[i] * (1 - alpha);

                overlayData[i + 1] =
                    overlayData[i + 1] * (1 - alpha);

                overlayData[i + 2] =
                    overlayData[i + 2] * (1 - alpha);
            }
        }

        context.putImageData(rgbData, 0, 0);

        return canvas.toDataURL("image/png");
    }

    async function loadMiningTiles() {
        const split = splitSelect.value;

        tileSelect.innerHTML = '<option value="">Loading tiles...</option>';
        tileSelect.disabled = true;
        analyzeButton.disabled = true;
        resultsElement.hidden = true;

        try {
            const response = await fetch(
                `${MINING_API_BASE_URL}/api/mining/tiles?split=${encodeURIComponent(split)}`
            );

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || "Unable to load mining tiles.");
            }

            const data = await response.json();

            tileSelect.innerHTML = '<option value="">Select a tile</option>';

            data.tile_ids.forEach((tileId) => {
                const option = document.createElement("option");
                option.value = tileId;
                option.textContent = `Tile ${tileId}`;
                tileSelect.appendChild(option);
            });

            tileSelect.disabled = false;

            setMiningStatus(
                `${data.count} tiles available in the ${split} split. Select a tile to continue.`,
                "success"
            );
        } catch (error) {
            tileSelect.innerHTML = '<option value="">No tiles available</option>';
            setMiningStatus(error.message, "error");
        }
    }

    async function displayMiningResults(data) {
        document.getElementById("miningCoverage").textContent =
            `${data.coverage_percentage}%`;

        document.getElementById("miningPixels").textContent =
            Number(data.predicted_pixels).toLocaleString();

        document.getElementById("miningThreshold").textContent =
            data.threshold;

        const images = [
            ["miningRgbImage", data.rgb_image],
            ["miningMaskImage", data.binary_mask],
            ["miningProbabilityImage", data.probability_map]
        ];

        images.forEach(([elementId, imageData]) => {
            const imageElement = document.getElementById(elementId);
            imageElement.src = imageData;
            imageElement.hidden = false;
        });

        const overlayImage = document.getElementById("miningOverlayImage");

        overlayImage.src = await createMiningOverlay(
            data.rgb_image,
            data.binary_mask
        );

        overlayImage.hidden = false;
        resultsElement.hidden = false;
    }

    async function analyzeMiningTile() {
        const split = splitSelect.value;
        const tileId = tileSelect.value;

        if (!tileId) {
            setMiningStatus("Please select a satellite tile first.", "error");
            return;
        }

        analyzeButton.disabled = true;
        resultsElement.hidden = true;
        setMiningStatus(`Analyzing tile ${tileId}. Please wait...`, "info");

        try {
            const response = await fetch(
                `${MINING_API_BASE_URL}/api/mining/analyze/${encodeURIComponent(split)}/${encodeURIComponent(tileId)}`
            );

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || "Mining analysis failed.");
            }

            const data = await response.json();

            await displayMiningResults(data);

            setMiningStatus(
                `Mining analysis completed successfully for tile ${tileId}.`,
                "success"
            );
        } catch (error) {
            setMiningStatus(error.message, "error");
        } finally {
            analyzeButton.disabled = false;
        }
    }

    splitSelect.addEventListener("change", loadMiningTiles);

    tileSelect.addEventListener("change", () => {
        analyzeButton.disabled = !tileSelect.value;
    });

    analyzeButton.addEventListener("click", analyzeMiningTile);

    loadMiningTiles();
});
