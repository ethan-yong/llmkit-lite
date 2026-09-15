const form = document.querySelector("#settings-form");
const saveButton = document.querySelector("#save-settings");
const testButton = document.querySelector("#test-connection");
const saveState = document.querySelector("#save-state");
const saveStateLabel = document.querySelector("#save-state-label");
const secretInput = document.querySelector("#api-key");
const secretState = document.querySelector("#secret-state");
const clearSecret = document.querySelector("#clear-api-key");
const toggleSecret = document.querySelector("#toggle-secret");
const providerSelect = document.querySelector("#provider-select");
const baseUrlInput = document.querySelector("#base-url");
const reasoningField = document.querySelector("#reasoning-field");
const connectionPanel = document.querySelector(".connection-panel");
const connectionMessage = document.querySelector("#connection-message");
const modelList = document.querySelector("#available-models");
const toast = document.querySelector("#toast");
const fields = [...document.querySelectorAll("[data-path]")];

const providerDefaults = {
  local: "http://127.0.0.1:4000",
  vllm: "http://127.0.0.1:8000",
  litellm: "http://127.0.0.1:4000",
  deepseek: "https://api.deepseek.com/v1",
};

let loading = true;
let apiKeyConfigured = false;
let toastTimer;

function endpoint(path) {
  return new URL(path, window.location.href).toString();
}

function getPath(source, path) {
  return path.split(".").reduce((value, part) => value?.[part], source);
}

function setPath(target, path, value) {
  const parts = path.split(".");
  const finalPart = parts.pop();
  const parent = parts.reduce((value, part) => {
    value[part] ??= {};
    return value[part];
  }, target);
  parent[finalPart] = value;
}

function fieldValue(field) {
  if (field.type === "checkbox") {
    return field.checked;
  }
  if (field.type === "number") {
    return Number(field.value);
  }
  return field.value || null;
}

function setFieldValue(field, value) {
  if (field.type === "checkbox") {
    field.checked = Boolean(value);
    return;
  }
  field.value = value ?? "";
}

function readForm() {
  const settings = { version: 1 };
  fields.forEach((field) => setPath(settings, field.dataset.path, fieldValue(field)));
  settings.clear_api_key = clearSecret.checked;
  return settings;
}

function hydrate(settings) {
  fields.forEach((field) => {
    setFieldValue(field, getPath(settings, field.dataset.path));
  });
  apiKeyConfigured = Boolean(settings.llm.api_key_configured);
  secretInput.value = "";
  clearSecret.checked = false;
  updateSecretState();
  updateProviderFields();
  updateTracingFields();
}

function setSaveState(state, label) {
  saveState.dataset.state = state;
  saveStateLabel.textContent = label;
}

function markDirty() {
  if (!loading) {
    setSaveState("dirty", "Unsaved changes");
  }
}

function updateSecretState() {
  const hasNewKey = Boolean(secretInput.value.trim());
  if (clearSecret.checked) {
    secretState.textContent = "Will be removed";
    secretState.classList.remove("configured");
    secretInput.disabled = true;
  } else if (hasNewKey) {
    secretState.textContent = "New key entered";
    secretState.classList.add("configured");
    secretInput.disabled = false;
  } else if (apiKeyConfigured) {
    secretState.textContent = "Saved locally";
    secretState.classList.add("configured");
    secretInput.disabled = false;
  } else {
    secretState.textContent = "Not configured";
    secretState.classList.remove("configured");
    secretInput.disabled = false;
  }
}

function updateProviderFields() {
  const isDeepSeek = providerSelect.value === "deepseek";
  const select = reasoningField.querySelector("select");
  select.disabled = isDeepSeek;
  reasoningField.style.opacity = isDeepSeek ? "0.5" : "1";
  if (isDeepSeek) {
    select.value = "";
  }
}

function updateTracingFields() {
  const enabled = document.querySelector(
    '[data-path="observability.tracing_enabled"]',
  ).checked;
  document.querySelector(".tracing-fields").classList.toggle("disabled", !enabled);
}

function showToast(message, isError = false) {
  clearTimeout(toastTimer);
  toast.textContent = message;
  toast.classList.toggle("error", isError);
  toast.classList.add("visible");
  toastTimer = setTimeout(() => toast.classList.remove("visible"), 3500);
}

function clearFieldErrors() {
  fields.forEach((field) => {
    field.removeAttribute("aria-invalid");
    field.removeAttribute("title");
  });
}

function showFieldErrors(fieldErrors = []) {
  fieldErrors.forEach(({ field: path, message }) => {
    const input = fields.find((field) => field.dataset.path === path);
    if (input) {
      input.setAttribute("aria-invalid", "true");
      input.title = message;
    }
  });
}

function validateTiming() {
  const settings = readForm();
  const heartbeat = settings.workflow.heartbeat_interval_seconds;
  if (
    heartbeat >= settings.workflow.lease_duration_seconds ||
    heartbeat >= settings.workflow.queue_visibility_seconds
  ) {
    showToast(
      "Heartbeat interval must be shorter than lease and queue visibility.",
      true,
    );
    return false;
  }
  return true;
}

async function parseResponse(response) {
  let payload;
  try {
    payload = await response.json();
  } catch {
    throw new Error("The settings service returned an unreadable response.");
  }
  if (!response.ok) {
    const error = new Error("Settings could not be saved.");
    error.payload = payload;
    throw error;
  }
  return payload;
}

async function loadSettings() {
  setSaveState("loading", "Loading settings");
  try {
    const response = await fetch(endpoint("api/settings"), {
      headers: { Accept: "application/json" },
    });
    const settings = await parseResponse(response);
    hydrate(settings);
    setSaveState("saved", "All changes saved");
  } catch (error) {
    setSaveState("error", "Could not load settings");
    showToast(error.message, true);
  } finally {
    loading = false;
  }
}

async function saveSettings({ quiet = false } = {}) {
  clearFieldErrors();
  if (!form.checkValidity()) {
    form.reportValidity();
    throw new Error("Check the highlighted settings.");
  }
  if (!validateTiming()) {
    throw new Error("Workflow timing is invalid.");
  }

  saveButton.disabled = true;
  setSaveState("loading", "Saving changes");
  try {
    const response = await fetch(endpoint("api/settings"), {
      method: "PUT",
      headers: { "Content-Type": "application/json", Accept: "application/json" },
      body: JSON.stringify(readForm()),
    });
    const settings = await parseResponse(response);
    hydrate(settings);
    setSaveState("saved", "All changes saved");
    if (!quiet) {
      showToast("Settings saved.");
    }
    return settings;
  } catch (error) {
    showFieldErrors(error.payload?.fields);
    setSaveState("error", "Save failed");
    if (!quiet) {
      showToast(error.message, true);
    }
    throw error;
  } finally {
    saveButton.disabled = false;
  }
}

async function testConnection() {
  testButton.disabled = true;
  connectionPanel.dataset.state = "loading";
  connectionMessage.textContent = "Checking the provider…";
  try {
    await saveSettings({ quiet: true });
    const response = await fetch(endpoint("api/test-connection"), {
      method: "POST",
      headers: { Accept: "application/json" },
    });
    const result = await parseResponse(response);
    connectionPanel.dataset.state = result.ok ? "success" : "error";
    connectionMessage.textContent = result.message;
    modelList.replaceChildren(
      ...result.models.map((model) => {
        const option = document.createElement("option");
        option.value = model;
        return option;
      }),
    );
    showToast(result.message, !result.ok);
  } catch (error) {
    connectionPanel.dataset.state = "error";
    connectionMessage.textContent = "Connection check failed.";
    showToast(error.message, true);
  } finally {
    testButton.disabled = false;
  }
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  try {
    await saveSettings();
  } catch {
    // The save helper already provides field-level and toast feedback.
  }
});

fields.forEach((field) => {
  field.addEventListener("input", () => {
    field.removeAttribute("aria-invalid");
    markDirty();
  });
});

secretInput.addEventListener("input", updateSecretState);
clearSecret.addEventListener("change", () => {
  updateSecretState();
  markDirty();
});

toggleSecret.addEventListener("click", () => {
  const show = secretInput.type === "password";
  secretInput.type = show ? "text" : "password";
  toggleSecret.textContent = show ? "Hide" : "Show";
  toggleSecret.setAttribute("aria-label", `${show ? "Hide" : "Show"} API key`);
});

providerSelect.addEventListener("change", () => {
  const knownDefault = Object.values(providerDefaults).includes(baseUrlInput.value);
  if (!baseUrlInput.value || knownDefault) {
    baseUrlInput.value = providerDefaults[providerSelect.value];
  }
  updateProviderFields();
  markDirty();
});

document
  .querySelector('[data-path="observability.tracing_enabled"]')
  .addEventListener("change", () => {
    updateTracingFields();
    markDirty();
  });

testButton.addEventListener("click", testConnection);

const observer = new IntersectionObserver(
  (entries) => {
    const visible = entries
      .filter((entry) => entry.isIntersecting)
      .sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];
    if (!visible) return;
    document.querySelectorAll(".nav-item").forEach((item) => {
      item.classList.toggle("active", item.hash === `#${visible.target.id}`);
    });
  },
  { rootMargin: "-15% 0px -65%", threshold: [0, 0.25, 0.5] },
);

document.querySelectorAll(".settings-section").forEach((section) => {
  observer.observe(section);
});

loadSettings();
