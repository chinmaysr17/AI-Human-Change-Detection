
// =====================================================
// AUTHENTICATION STATE
// =====================================================

window.currentUser = null;


// =====================================================
// AUTHENTICATION STATUS
// =====================================================

function showAuthStatus(message, isError = false) {
    const status = document.getElementById("authStatus");

    if (!status) return;

    status.textContent = message;
    status.className = isError
        ? "auth-status error"
        : "auth-status";
}
function showAuthView(view) {
    const loginCard = document.getElementById("loginCard");
    const registerCard = document.getElementById("registerCard");
    const status = document.getElementById("authStatus");

    if (!loginCard || !registerCard) return;

    loginCard.hidden = view !== "login";
    registerCard.hidden = view !== "register";

    if (status) {
        status.textContent = "";
        status.className = "auth-status";
    }
}

// =====================================================
// DASHBOARD NAVIGATION
// =====================================================

const dashboardPageTitles = {
    dashboardOverview: "Dashboard",
    locationSection: "Satellite Analysis",
    historySection: "Analysis History",
    modelSection: "Model Information",
    settingsSection: "Settings"
};


function showDashboardPage(pageId) {
    const validPageIds = Object.keys(dashboardPageTitles);

    if (!validPageIds.includes(pageId)) {
        return;
    }

    // Hide all dashboard pages.
    validPageIds.forEach(id => {
        const page = document.getElementById(id);

        if (page) {
            page.hidden = id !== pageId;
        }
    });

    // Update the page heading.
    const pageTitle = document.getElementById("dashboardPageTitle");

    if (pageTitle) {
        pageTitle.textContent = dashboardPageTitles[pageId];
    }

    // Update active sidebar navigation.
    document.querySelectorAll(".sidebar-link").forEach(button => {
        const isActive = button.dataset.pageTarget === pageId;

        button.classList.toggle("active", isActive);

        if (isActive) {
            button.setAttribute("aria-current", "page");
        } else {
            button.removeAttribute("aria-current");
        }
    });

    // Recalculate Leaflet map dimensions when its page is displayed.
    if (pageId === "locationSection") {
        requestAnimationFrame(() => {
            window.dispatchEvent(new Event("resize"));
        });
    }
}


// =====================================================
// INITIALIZE DASHBOARD NAVIGATION
// =====================================================

function initializeDashboardNavigation() {
    const navigationButtons = document.querySelectorAll(
        "[data-page-target]"
    );

    navigationButtons.forEach(button => {
        button.addEventListener("click", () => {
            const targetPage = button.dataset.pageTarget;

            showDashboardPage(targetPage);
        });
    });
}


// =====================================================
// AUTHENTICATION UI
// =====================================================

function updateAuthenticationUI(user) {
    const authSection = document.getElementById("authSection");
    const authForms = document.getElementById("authForms");
    const userPanel = document.getElementById("userPanel");

    const dashboardShell = document.getElementById("dashboardShell");

    const userName = document.getElementById("loggedInUserName");
    const userEmail = document.getElementById("loggedInUserEmail");

    const dashboardUserName = document.getElementById(
        "dashboardUserName"
    );

    window.currentUser = user || null;

    if (user) {

        // Hide authentication and reveal dashboard.
        if (authSection) authSection.hidden = true;
        if (dashboardShell) dashboardShell.hidden = false;

        if (authForms) authForms.hidden = true;
        if (userPanel) userPanel.hidden = false;

        // Update authenticated user details.
        if (userName) {
            userName.textContent = user.full_name || "";
        }

        if (userEmail) {
            userEmail.textContent = user.email || "";
        }

        if (dashboardUserName) {
            dashboardUserName.textContent =
                user.full_name || user.email || "User";
        }
        // Update Settings Profile Information
const profileName = document.getElementById("settingsProfileName");
const profileEmail = document.getElementById("settingsProfileEmail");
const profileAvatar = document.getElementById("settingsProfileAvatar");
const fullName = document.getElementById("settingsFullName");
const emailAddress = document.getElementById("settingsEmail");

const name = user.full_name || user.email || "User";
const email = user.email || "Not available";

if (profileName) profileName.textContent = name;
if (profileEmail) profileEmail.textContent = email;
if (profileAvatar) profileAvatar.textContent = name.charAt(0).toUpperCase();
if (fullName) fullName.textContent = user.full_name || "Not available";
if (emailAddress) emailAddress.textContent = email;
        // Start at dashboard overview.
        showDashboardPage("dashboardOverview");

        // Refresh map dimensions after dashboard becomes visible.
        requestAnimationFrame(() => {
            window.dispatchEvent(new Event("resize"));
        });

        showAuthStatus("You are logged in successfully.");

        // Load only this authenticated user's history.
        if (typeof loadAnalysisHistory === "function") {
            loadAnalysisHistory();
        }

    } else {

        // Hide dashboard and reveal authentication.
        if (dashboardShell) dashboardShell.hidden = true;
        if (authSection) authSection.hidden = false;

        if (authForms) authForms.hidden = false;
        if (userPanel) userPanel.hidden = true;

        // Clear user details.
        if (userName) userName.textContent = "";
        if (userEmail) userEmail.textContent = "";

        if (dashboardUserName) {
            dashboardUserName.textContent = "User";
        }

        // Reset dashboard to its default page.
        showDashboardPage("dashboardOverview");

        // Clear authenticated user's history.
        if (typeof clearAnalysisHistory === "function") {
            clearAnalysisHistory();
        }

        showAuthStatus("Please log in or create an account.");
    }
}


// =====================================================
// LOGIN
// =====================================================

async function handleLogin(event) {
    event.preventDefault();

    const email = document.getElementById("loginEmail").value.trim();
    const password = document.getElementById("loginPassword").value;

    const button = document.getElementById("loginButton");

    if (button) {
        button.disabled = true;
        button.textContent = "Logging in...";
    }

    try {
        const result = await loginUser({
            email,
            password
        });

        const user = result.user;

        if (!user || !user.id) {
            throw new Error(
                "Invalid user information received from backend."
            );
        }

        updateAuthenticationUI(user);

        document.getElementById("loginForm").reset();

        showAuthStatus(`Welcome back, ${user.full_name}!`);

    } catch (error) {
        console.error("Login failed:", error);
        showAuthStatus(error.message, true);

    } finally {
        if (button) {
            button.disabled = false;
            button.textContent = "Sign In";
        }
    }
}


// =====================================================
// REGISTRATION
// =====================================================

async function handleRegistration(event) {
    event.preventDefault();

    const fullName = document.getElementById("registerName").value.trim();
    const email = document.getElementById("registerEmail").value.trim();
    const password = document.getElementById("registerPassword").value;

    const button = document.getElementById("registerButton");

    if (button) {
        button.disabled = true;
        button.textContent = "Registering...";
    }

    try {
        await registerUser({
            full_name: fullName,
            email,
            password
        });

        document.getElementById("registerForm").reset();

        showAuthStatus(
            "Registration successful! Please log in using your new account."
        );

        // Fill login email for convenience.
        document.getElementById("loginEmail").value = email;

    } catch (error) {
        console.error("Registration failed:", error);
        showAuthStatus(error.message, true);

    } finally {
        if (button) {
            button.disabled = false;
            button.textContent = "Create Account";
        }
    }
}


// =====================================================
// LOGOUT
// =====================================================

async function handleLogout() {
    const button = document.getElementById("logoutButton");

    if (button) {
        button.disabled = true;
        button.textContent = "Logging out...";
    }

    try {
        await logoutUser();

        updateAuthenticationUI(null);

        document.getElementById("loginForm")?.reset();
        document.getElementById("registerForm")?.reset();

        showAuthStatus("You have logged out successfully.");

    } catch (error) {
        console.error("Logout failed:", error);

        // Do not clear local authentication state if logout
        // has not been confirmed by the backend.
        showAuthStatus(error.message, true);

    } finally {
        if (button) {
            button.disabled = false;
            button.textContent = "Logout";
        }
    }
}


// =====================================================
// RESTORE EXISTING SESSION
// =====================================================

async function restoreAuthenticationSession() {
    showAuthStatus("Checking your login session...");

    try {
        const user = await getCurrentUser();

        if (!user || !user.id) {
            throw new Error("Invalid session information.");
        }

        updateAuthenticationUI(user);

    } catch (error) {
        updateAuthenticationUI(null);
    }
}


// =====================================================
// REGISTER AUTHENTICATION EVENTS
// =====================================================

document.addEventListener("DOMContentLoaded", () => {

    const loginForm = document.getElementById("loginForm");
    const registerForm = document.getElementById("registerForm");
    const logoutButton = document.getElementById("logoutButton");

    if (loginForm) {
        loginForm.addEventListener("submit", handleLogin);
    }

    if (registerForm) {
        registerForm.addEventListener("submit", handleRegistration);
    }

    if (logoutButton) {
        logoutButton.addEventListener("click", handleLogout);
    }
    document.getElementById("showRegisterButton")
    ?.addEventListener("click", () => showAuthView("register"));

document.getElementById("showLoginButton")
    ?.addEventListener("click", () => showAuthView("login"));

showAuthView("login");

// =========================================
// APPEARANCE SETTINGS - THEME MANAGEMENT
// =========================================

const themeToggle = document.getElementById("themeToggle");
const themeModeLabel = document.getElementById("themeModeLabel");
const themeModeDescription = document.getElementById(
    "themeModeDescription"
);

function applyTheme(theme) {
    const selectedTheme = theme === "dark" ? "dark" : "light";

    document.body.dataset.theme = selectedTheme;

    if (themeToggle) {
        themeToggle.checked = selectedTheme === "dark";
    }

    if (themeModeLabel) {
        themeModeLabel.textContent =
            selectedTheme === "dark" ? "Dark Mode" : "Light Mode";
    }

    if (themeModeDescription) {
        themeModeDescription.textContent =
            selectedTheme === "dark"
                ? "Comfortable low-light appearance."
                : "Clean and bright appearance.";
    }

    try {
        localStorage.setItem("geovision-theme", selectedTheme);
    } catch (error) {
        console.warn("Could not save theme preference:", error);
    }
}

let savedTheme = "light";

try {
    savedTheme = localStorage.getItem("geovision-theme") || "light";
} catch (error) {
    console.warn("Could not read theme preference:", error);
}

applyTheme(savedTheme);

themeToggle?.addEventListener("change", () => {
    applyTheme(themeToggle.checked ? "dark" : "light");
});


// =========================================
// SETTINGS PAGE NAVIGATION
// =========================================

const settingsNavButtons = document.querySelectorAll(
    ".settings-nav-item[data-settings-target]"
);

const settingsPanels = document.querySelectorAll(
    ".settings-content > .settings-card"
);

settingsNavButtons.forEach(button => {
    button.addEventListener("click", () => {
        const targetId = button.dataset.settingsTarget;

        settingsPanels.forEach(panel => {
            panel.hidden = panel.id !== targetId;
        });

        settingsNavButtons.forEach(navButton => {
            navButton.classList.toggle(
                "active",
                navButton === button
            );
        });
    });
});

// Show Profile when Settings is opened
const settingsSidebarButton = document.querySelector(
    '[data-page-target="settingsSection"]'
);

settingsSidebarButton?.addEventListener("click", () => {
    settingsNavButtons.forEach(button => {
        const isProfile = button.dataset.settingsTarget === "profileSettings";

        button.classList.toggle("active", isProfile);
    });

    settingsPanels.forEach(panel => {
        panel.hidden = panel.id !== "profileSettings";
    });
});


    // Initialize sidebar and dashboard navigation.
    initializeDashboardNavigation();

    // Restore login state when the page opens or refreshes.
    restoreAuthenticationSession();
});
