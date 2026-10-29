
// =====================================================
// MAIN.JS
// Main page initialization and model information
// =====================================================


// =====================================================
// DISPLAY MODEL INFORMATION
// =====================================================

async function loadModelInformation() {
    try {
        const data = await getModelInformation();

        const model = data.model || {};
        const input = data.input || {};
        const dataset = data.dataset || {};
        const evaluation = data.evaluation || {};
        const scope = data.scope || {};

        const setText = (elementId, value) => {
            const element = document.getElementById(elementId);

            if (element) {
                element.textContent =
                    value === undefined ||
                    value === null ||
                    value === ""
                        ? "--"
                        : String(value);
            }
        };

        setText("modelName", model.name);
        setText("modelArchitecture", model.architecture);
        setText("modelTask", model.task);

        const parameterCount = model.parameters;

        setText(
            "modelParameters",
            typeof parameterCount === "number"
                ? parameterCount.toLocaleString()
                : parameterCount
        );

        const inputShape = input.shape;

        setText(
            "modelInput",
            Array.isArray(inputShape)
                ? inputShape.join(" × ")
                : inputShape
        );

        setText("modelDataset", dataset.name);
        setText("modelThreshold", evaluation.threshold);

        const f1Score = evaluation.test_f1_percent;

        setText(
            "modelF1",
            f1Score === undefined || f1Score === null
                ? "--"
                : `${f1Score}%`
        );

        setText("modelNote", scope.note);

        console.log("Model information loaded successfully.");
    } catch (error) {
        console.error("Failed to load model information:", error);

        const note = document.getElementById("modelNote");

        if (note) {
            note.textContent =
                "Model information could not be loaded. " +
                "Please check that the backend is running.";
        }
    }
}


// =====================================================
// INITIALIZE THE APPLICATION
// =====================================================

function initializeApplication() {
    console.log(
        "AI-Based Human-Induced Change Detection initialized."
    );

    loadModelInformation();
}


// =====================================================
// START WHEN THE PAGE IS READY
// =====================================================

document.addEventListener(
    "DOMContentLoaded",
    initializeApplication
);
