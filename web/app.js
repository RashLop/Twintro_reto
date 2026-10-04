const profileForm = document.querySelector("#profile-form");
const matchForm = document.querySelector("#match-form");
const sourceSelect = document.querySelector("#source-profile");
const candidateSelect = document.querySelector("#candidate-profile");
const candidateList = document.querySelector("#candidate-list");
const singleCandidateWrap = document.querySelector("#single-candidate-wrap");
const multipleCandidatesWrap = document.querySelector("#multiple-candidates-wrap");
const matchButton = document.querySelector("#match-submit");
const messageBox = document.querySelector("#app-message");
const resultsSection = document.querySelector("#results-section");
const resultsList = document.querySelector("#results-list");
const resultsMeta = document.querySelector("#results-meta");
const savedProfileNote = document.querySelector("#profile-saved");
const toggleCandidatesButton = document.querySelector("#toggle-candidates");
const skillPicker = document.querySelector('[data-picker="skills"]');
const interestPicker = document.querySelector('[data-picker="interests"]');

let profiles = [];
let isSavingProfile = false;
let isMatching = false;
const pickerLimits = { skills: 8, interests: 5 };

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (character) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;",
  })[character]);
}

function showMessage(text, isError = false) {
  messageBox.textContent = text;
  messageBox.classList.toggle("error", isError);
  messageBox.hidden = false;
}

function clearMessage() {
  messageBox.hidden = true;
  messageBox.textContent = "";
  messageBox.classList.remove("error");
}

async function requestJson(url, options = {}) {
  const response = await fetch(url, options);
  let body;
  try {
    body = await response.json();
  } catch {
    throw new Error("La API devolvió una respuesta que no es JSON válido.");
  }
  if (!response.ok) {
    throw new Error(body.error || `La solicitud falló (${response.status}).`);
  }
  return body;
}

function displayName(profile) {
  const data = profile.professional_profile || {};
  return data.name || data.role || profile.user_id;
}

function displayRole(profile) {
  const data = profile.professional_profile || {};
  return [data.role, data.industry].filter(Boolean).join(" · ") || "Perfil profesional";
}

function updateMatchButton() {
  const sourceId = sourceSelect.value;
  const mode = matchForm.elements["match-mode"].value;
  let canMatch = Boolean(sourceId) && profiles.some((profile) => profile.user_id === sourceId);
  if (mode === "one-to-one") {
    canMatch = canMatch && Boolean(candidateSelect.value) && sourceId !== candidateSelect.value;
  } else {
    canMatch = canMatch && candidateList.querySelectorAll('input[type="checkbox"]:checked').length > 0;
  }
  matchButton.disabled = !canMatch || isMatching;
  matchButton.querySelector("span:first-child").textContent = isMatching ? "Buscando matches..." : "Encontrar matches";
}

function availableCandidates() {
  return profiles.filter((profile) => profile.user_id !== sourceSelect.value);
}

function renderSourceOptions(preferredId = sourceSelect.value) {
  const previous = preferredId;
  sourceSelect.innerHTML = '<option value="">Selecciona tu perfil</option>';
  for (const profile of profiles) {
    const option = document.createElement("option");
    option.value = profile.user_id;
    option.textContent = `${displayName(profile)} — ${displayRole(profile)}`;
    sourceSelect.append(option);
  }
  if (profiles.some((profile) => profile.user_id === previous)) {
    sourceSelect.value = previous;
  }
}

function renderCandidates() {
  const candidates = availableCandidates();
  const currentCandidate = candidateSelect.value;
  candidateSelect.innerHTML = '<option value="">Selecciona un perfil</option>';
  for (const profile of candidates) {
    const option = document.createElement("option");
    option.value = profile.user_id;
    option.textContent = `${displayName(profile)} — ${displayRole(profile)}`;
    candidateSelect.append(option);
  }
  if (candidates.some((profile) => profile.user_id === currentCandidate)) {
    candidateSelect.value = currentCandidate;
  }

  candidateList.replaceChildren();
  if (candidates.length === 0) {
    const empty = document.createElement("div");
    empty.className = "empty-candidates";
    empty.textContent = "No hay otros perfiles todavía. Crea uno para empezar.";
    candidateList.append(empty);
  } else {
    for (const profile of candidates) {
      const label = document.createElement("label");
      label.className = "candidate-check";
      const checkbox = document.createElement("input");
      checkbox.type = "checkbox";
      checkbox.value = profile.user_id;
      const text = document.createElement("span");
      text.append(document.createTextNode(displayName(profile)));
      const detail = document.createElement("small");
      detail.textContent = displayRole(profile);
      text.append(detail);
      label.append(checkbox, text);
      candidateList.append(label);
    }
  }
  updateMatchButton();
}

async function loadProfiles(preferredSourceId) {
  const response = await requestJson("/api/profiles");
  profiles = response.profiles;
  renderSourceOptions(preferredSourceId);
  renderCandidates();
  return profiles;
}

function renderSelectOptions(select, values, placeholder, selectedValue = "") {
  select.replaceChildren(new Option(placeholder, ""));
  for (const value of values) {
    select.add(new Option(value, value));
  }
  if (values.includes(selectedValue)) select.value = selectedValue;
}

function selectedPickerValues(picker) {
  return Array.from(picker.querySelectorAll('input[type="checkbox"]:checked'), (input) => input.value);
}

function updatePickerCount(picker) {
  const selectedCount = selectedPickerValues(picker).length;
  const limit = pickerLimits[picker.dataset.picker];
  const count = picker.querySelector("[data-picker-count]");
  count.textContent = `${selectedCount} seleccionad${selectedCount === 1 ? "a" : "as"}`;
  for (const checkbox of picker.querySelectorAll('input[type="checkbox"]:not(:checked)')) {
    checkbox.disabled = selectedCount >= limit;
  }
}

function renderPicker(picker, values) {
  const options = picker.querySelector(".picker-options");
  options.replaceChildren();
  for (const value of values) {
    const label = document.createElement("label");
    label.className = "picker-option";
    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.value = value;
    const text = document.createElement("span");
    text.textContent = value;
    label.append(checkbox, text);
    options.append(label);
  }
  updatePickerCount(picker);
}

async function loadProfileOptions() {
  const options = await requestJson("/api/options");
  renderSelectOptions(
    profileForm.elements.role,
    options.roles,
    "Selecciona tu rol",
  );
  renderSelectOptions(
    profileForm.elements.industry,
    options.industries,
    "Selecciona tu industria",
  );
  renderSelectOptions(
    profileForm.elements.experience_years,
    options.experience_years.map(String),
    "Selecciona años",
  );
  renderPicker(skillPicker, options.skills);
  renderPicker(interestPicker, options.interests);
}

function filterPicker(picker) {
  const search = picker.querySelector(".picker-search").value.trim().toLocaleLowerCase();
  for (const option of picker.querySelectorAll(".picker-option")) {
    option.hidden = !option.textContent.toLocaleLowerCase().includes(search);
  }
}

profileForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  clearMessage();
  savedProfileNote.hidden = true;
  if (!profileForm.reportValidity() || isSavingProfile) return;

  const skills = selectedPickerValues(skillPicker);
  const interests = selectedPickerValues(interestPicker);
  if (skills.length === 0) {
    showMessage("Selecciona al menos una habilidad.", true);
    return;
  }

  const formData = new FormData(profileForm);
  const profile = {
    professional_profile: {
      name: formData.get("name").trim(),
      role: formData.get("role"),
      industry: formData.get("industry"),
      skills,
      experience_years: Number(formData.get("experience_years")),
      interests,
    },
  };

  isSavingProfile = true;
  const submitButton = profileForm.querySelector('button[type="submit"]');
  submitButton.disabled = true;
  submitButton.querySelector("span:first-child").textContent = "Guardando perfil...";
  try {
    const result = await requestJson("/api/profiles", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(profile),
    });
    await loadProfiles(result.profile.user_id);
    savedProfileNote.textContent = `Perfil guardado. Tu ID es ${result.profile.user_id}; ya está seleccionado para comparar.`;
    savedProfileNote.hidden = false;
    profileForm.reset();
    for (const picker of [skillPicker, interestPicker]) {
      for (const checkbox of picker.querySelectorAll('input[type="checkbox"]')) {
        checkbox.checked = false;
      }
      picker.querySelector(".picker-search").value = "";
      filterPicker(picker);
      updatePickerCount(picker);
    }
    profileForm.elements.experience_years.value = "";
    showMessage("¡Perfil creado! Ahora puedes compararlo con cualquier perfil de la lista.");
  } catch (error) {
    showMessage(error.message, true);
  } finally {
    isSavingProfile = false;
    submitButton.disabled = false;
    submitButton.querySelector("span:first-child").textContent = "Guardar mi perfil";
    updateMatchButton();
  }
});

function selectedMode() {
  return matchForm.querySelector('input[name="match-mode"]:checked').value;
}

function updateMode() {
  const oneToOne = selectedMode() === "one-to-one";
  singleCandidateWrap.hidden = !oneToOne;
  multipleCandidatesWrap.hidden = oneToOne;
  candidateSelect.required = oneToOne;
  updateMatchButton();
}

function renderResults(matches) {
  resultsList.replaceChildren();
  if (matches.length === 0) {
    const empty = document.createElement("div");
    empty.className = "empty-candidates";
    empty.textContent = "No encontramos perfiles para comparar. Revisa los perfiles seleccionados.";
    resultsList.append(empty);
    resultsSection.hidden = false;
    return;
  }
  for (const match of matches) {
    const target = match.target_user;
    const comparison = match.profile_comparison || {};
    const card = document.createElement("article");
    card.className = "result-card";
    const initials = displayInitials(target.name || target.user_id);
    const tags = (comparison.shared_domains || [])
      .map((domain) => `<span class="tag">${escapeHtml(domain)}</span>`)
      .join("");
    const strengths = (comparison.complementary_strengths || [])
      .map((strength) => `<p class="result-copy">${escapeHtml(strength)}</p>`)
      .join("");
    card.innerHTML = `
      <div class="result-top">
        <div class="avatar" aria-hidden="true">${escapeHtml(initials)}</div>
        <div class="result-identity">
          <strong>${escapeHtml(target.name || "Profesional")}</strong>
          <span>${escapeHtml([target.role, target.industry].filter(Boolean).join(" · ") || target.user_id)}</span>
        </div>
        <div class="affinity">${escapeHtml(match.affinity_percentage)}%<span class="affinity-label">afinidad</span></div>
      </div>
      <div class="progress-track" role="img" aria-label="${escapeHtml(match.affinity_percentage)}% de afinidad">
        <div class="progress-fill" style="width:${safePercentage(match.affinity_percentage)}%"></div>
      </div>
      ${tags ? `<p class="result-section-label">En común</p><div class="tag-list">${tags}</div>` : ""}
      ${strengths ? `<p class="result-section-label">Lo que pueden aportar</p>${strengths}` : ""}
      <p class="result-section-label">Por qué conectan</p>
      <p class="result-copy">${escapeHtml(match.justification || "Afinidad calculada a partir de los perfiles profesionales.")}</p>
    `;
    resultsList.append(card);
  }
  resultsSection.hidden = false;
}

function safePercentage(value) {
  const percentage = Number(value);
  return Number.isFinite(percentage) ? Math.max(0, Math.min(100, percentage)) : 0;
}

function displayInitials(name) {
  return name.trim().split(/\s+/).slice(0, 2).map((part) => part[0]?.toUpperCase() || "").join("") || "T";
}

matchForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  clearMessage();
  if (!matchForm.reportValidity() || isMatching) return;
  const mode = selectedMode();
  const sourceId = sourceSelect.value;
  const payload = { match_mode: mode === "one-to-one" ? "1:1" : "1:N", source_id: sourceId };
  let endpoint;
  if (mode === "one-to-one") {
    endpoint = "/api/matches/one-to-one";
    payload.candidate_id = candidateSelect.value;
  } else {
    endpoint = "/api/matches/one-to-many";
    payload.candidate_ids = Array.from(candidateList.querySelectorAll('input[type="checkbox"]:checked'), (input) => input.value);
    payload.top_k = Number(document.querySelector("#top-k").value);
  }

  isMatching = true;
  updateMatchButton();
  try {
    const response = await requestJson(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    renderResults(response.ranked_matches);
    resultsMeta.textContent = `${response.ranked_matches.length} resultado${response.ranked_matches.length === 1 ? "" : "s"} · ${Number(response.api_latency_ms).toFixed(1)} ms`;
    resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (error) {
    showMessage(error.message, true);
  } finally {
    isMatching = false;
    updateMatchButton();
  }
});

sourceSelect.addEventListener("change", () => {
  renderCandidates();
  resultsSection.hidden = true;
});
candidateSelect.addEventListener("change", updateMatchButton);
candidateList.addEventListener("change", updateMatchButton);
matchForm.addEventListener("change", (event) => {
  if (event.target.name === "match-mode") updateMode();
});
toggleCandidatesButton.addEventListener("click", () => {
  const boxes = Array.from(candidateList.querySelectorAll('input[type="checkbox"]'));
  const selectAll = boxes.some((checkbox) => !checkbox.checked);
  for (const checkbox of boxes) checkbox.checked = selectAll;
  toggleCandidatesButton.textContent = selectAll ? "Quitar selección" : "Seleccionar todos";
  updateMatchButton();
});

for (const picker of [skillPicker, interestPicker]) {
  picker.addEventListener("change", (event) => {
    if (event.target.matches('input[type="checkbox"]')) {
      const selected = selectedPickerValues(picker);
      if (selected.length > pickerLimits[picker.dataset.picker]) {
        event.target.checked = false;
        showMessage(`Puedes seleccionar hasta ${pickerLimits[picker.dataset.picker]} opciones.`, true);
      } else {
        clearMessage();
      }
      updatePickerCount(picker);
    }
  });
  picker.querySelector(".picker-search").addEventListener("input", () => filterPicker(picker));
}

Promise.all([loadProfiles(), loadProfileOptions()]).catch((error) => {
  showMessage(`No se pudieron cargar los datos: ${error.message}`, true);
});
