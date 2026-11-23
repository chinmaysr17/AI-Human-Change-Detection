
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


// =====================================================
// AUTHENTICATION UI
// =====================================================

function updateAuthenticationUI(user) {
    const authForms = document.getElementById("authForms");
    const userPanel = document.getElementById("userPanel");
    const userName = document.getElementById("loggedInUserName");
    const userEmail = document.getElementById("loggedInUserEmail");

    window.currentUser = user || null;

    if (user) {
        if (authForms) authForms.hidden = true;
        if (userPanel) userPanel.hidden = false;

        if (userName) userName.textContent = user.full_name || "";
        if (userEmail) userEmail.textContent = user.email || "";

        showAuthStatus("You are logged in successfully.");

        // Load only this authenticated user's history.
        if (typeof loadAnalysisHistory === "function") {
            loadAnalysisHistory();
        }

    } else {
        if (authForms) authForms.hidden = false;
        if (userPanel) userPanel.hidden = true;

        if (userName) userName.textContent = "";
        if (userEmail) userEmail.textContent = "";

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
            throw new Error("Invalid user information received from backend.");
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
            button.textContent = "Login";
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
            button.textContent = "Register";
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

    // Restore login state when the page opens or refreshes.
    restoreAuthenticationSession();
});
