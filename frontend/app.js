const elements = {
  apiUrl: document.querySelector("#api-url"),
  userId: document.querySelector("#user-id"),
  sessionId: document.querySelector("#session-id"),
  topK: document.querySelector("#top-k"),
  scoreThreshold: document.querySelector("#score-threshold"),
  recipientId: document.querySelector("#recipient-id"),
  recipientName: document.querySelector("#recipient-name"),
  useDatabaseShortTerm: document.querySelector("#use-database-short-term"),
  form: document.querySelector("#chat-form"),
  messageInput: document.querySelector("#message-input"),
  sendButton: document.querySelector("#send-button"),
  messages: document.querySelector("#messages"),
  progressList: document.querySelector("#progress-list"),
  timingsOutput: document.querySelector("#timings-output"),
  specialistOutput: document.querySelector("#specialist-output"),
  routePill: document.querySelector("#route-pill"),
  healthDot: document.querySelector("#health-dot"),
  healthLabel: document.querySelector("#health-label"),
  healthDetail: document.querySelector("#health-detail"),
  resetSession: document.querySelector("#reset-session"),
};

const storageKey = "kapruka-frontend-settings";

function loadSettings() {
  const saved = JSON.parse(localStorage.getItem(storageKey) || "{}");
  for (const [key, value] of Object.entries(saved)) {
    if (!elements[key]) continue;
    if (elements[key].type === "checkbox") {
      elements[key].checked = Boolean(value);
    } else {
      elements[key].value = value;
    }
  }
}

function saveSettings() {
  const settings = {
    apiUrl: elements.apiUrl.value,
    userId: elements.userId.value,
    sessionId: elements.sessionId.value,
    topK: elements.topK.value,
    scoreThreshold: elements.scoreThreshold.value,
    recipientId: elements.recipientId.value,
    recipientName: elements.recipientName.value,
    useDatabaseShortTerm: elements.useDatabaseShortTerm.checked,
  };
  localStorage.setItem(storageKey, JSON.stringify(settings));
}

function apiBase() {
  return elements.apiUrl.value.replace(/\/+$/, "");
}

async function checkHealth() {
  const base = apiBase();
  elements.healthDetail.textContent = base;
  elements.healthDot.className = "status-dot";
  elements.healthLabel.textContent = "Checking API";

  try {
    const response = await fetch(`${base}/health`);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    elements.healthDot.className = "status-dot ok";
    elements.healthLabel.textContent = "API connected";
  } catch (error) {
    elements.healthDot.className = "status-dot bad";
    elements.healthLabel.textContent = "API offline";
    elements.healthDetail.textContent = `${base} (${error.message})`;
  }
}

function appendMessage(role, text, options = {}) {
  const article = document.createElement("article");
  article.className = `message ${role}`;
  if (options.pending) {
    article.classList.add("pending");
  }

  const avatar = document.createElement("div");
  avatar.className = "avatar";
  avatar.textContent = role === "user" ? "You" : "K";

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.innerHTML = options.pending ? renderPending() : renderAnswer(text);

  article.append(avatar, bubble);
  elements.messages.append(article);
  elements.messages.scrollTop = elements.messages.scrollHeight;
  return article;
}

function renderPending() {
  return `
    <div class="typing" aria-label="Assistant is thinking">
      <span></span>
      <span></span>
      <span></span>
    </div>
  `;
}

function renderAnswer(text) {
  const productLinks = [];
  const withoutMarkdownLinks = text.replace(
    /\[([^\]]+)\]\((https?:\/\/[^)\s]+)\)/g,
    (_match, _label, url) => {
      productLinks.push(cleanUrl(url));
      return "";
    },
  );
  const withoutRawLinks = withoutMarkdownLinks.replace(/https?:\/\/[^\s)]+/g, (url) => {
    productLinks.push(cleanUrl(url));
    return "";
  });
  const cleaned = cleanAssistantText(withoutRawLinks);
  const paragraphs = escapeHtml(cleaned)
    .split(/\n{2,}/)
    .map((paragraph) => paragraph.trim())
    .filter(Boolean)
    .map((paragraph) => `<p>${paragraph.replace(/\n/g, "<br>")}</p>`)
    .join("");

  const uniqueLinks = [...new Set(productLinks)].filter(Boolean);
  if (!uniqueLinks.length) {
    return paragraphs || "<p>No answer returned.</p>";
  }

  const buttons = uniqueLinks
    .map(
      (url, index) =>
        `<a class="product-button" href="${escapeAttribute(url)}" target="_blank" rel="noreferrer">View product${uniqueLinks.length > 1 ? ` ${index + 1}` : ""}</a>`,
    )
    .join("");
  return `${paragraphs}<div class="product-actions">${buttons}</div>`;
}

function cleanAssistantText(text) {
  return text
    .replace(/\*\*/g, "")
    .replace(/^\s*[-*]\s*/gm, "")
    .replace(/\[View Product\]\(\s*\)/gi, "")
    .replace(/\bLink:\s*$/gim, "")
    .replace(/[ \t]+\n/g, "\n")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}

function cleanUrl(url) {
  return url
    .replace(/^\[/, "")
    .replace(/\]$/, "")
    .replace(/[),.]+$/, "");
}

function escapeHtml(value) {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function escapeAttribute(value) {
  return escapeHtml(value);
}

function setProgress(items) {
  elements.progressList.innerHTML = "";
  if (!items || items.length === 0) {
    const item = document.createElement("li");
    item.textContent = "No progress messages.";
    elements.progressList.append(item);
    return;
  }

  for (const progress of items) {
    const item = document.createElement("li");
    item.textContent = progress;
    elements.progressList.append(item);
  }
}

function setDebug(data) {
  elements.routePill.textContent = data.route || "unknown";
  setProgress(data.progress || []);
  elements.timingsOutput.textContent = JSON.stringify(data.timings_ms || {}, null, 2);
  elements.specialistOutput.textContent = JSON.stringify(data.specialist_output || {}, null, 2);
}

function payload(message) {
  return {
    message,
    user_id: elements.userId.value.trim() || "demo-user",
    session_id: elements.sessionId.value.trim() || "demo-session",
    recipient_id: elements.recipientId.value.trim() || null,
    recipient_name: elements.recipientName.value.trim(),
    top_k: Number(elements.topK.value || 5),
    score_threshold: Number(elements.scoreThreshold.value || 0),
    use_database_short_term: elements.useDatabaseShortTerm.checked,
  };
}

async function sendMessage(message) {
  saveSettings();
  appendMessage("user", message);
  const pendingMessage = appendMessage("assistant", "", { pending: true });
  elements.sendButton.disabled = true;
  elements.sendButton.textContent = "Sending";
  setProgress(["Sending request..."]);

  try {
    const response = await fetch(`${apiBase()}/chat`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload(message)),
    });

    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || `HTTP ${response.status}`);
    }

    pendingMessage.remove();
    appendMessage("assistant", data.answer || "No answer returned.");
    setDebug(data);
  } catch (error) {
    pendingMessage.remove();
    appendMessage("assistant", `Request failed: ${error.message}`);
    setProgress(["Request failed. Check the API URL and backend server."]);
    elements.routePill.textContent = "error";
  } finally {
    elements.sendButton.disabled = false;
    elements.sendButton.textContent = "Send";
    elements.messageInput.focus();
  }
}

elements.form.addEventListener("submit", (event) => {
  event.preventDefault();
  const message = elements.messageInput.value.trim();
  if (!message) return;
  elements.messageInput.value = "";
  sendMessage(message);
});

elements.messageInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    elements.form.requestSubmit();
  }
});

document.querySelectorAll("[data-example]").forEach((button) => {
  button.addEventListener("click", () => {
    elements.messageInput.value = button.dataset.example;
    elements.messageInput.focus();
  });
});

elements.resetSession.addEventListener("click", () => {
  elements.sessionId.value = `session-${Date.now()}`;
  saveSettings();
});

[
  elements.apiUrl,
  elements.userId,
  elements.sessionId,
  elements.topK,
  elements.scoreThreshold,
  elements.recipientId,
  elements.recipientName,
  elements.useDatabaseShortTerm,
].forEach((input) => input.addEventListener("change", saveSettings));

elements.apiUrl.addEventListener("change", checkHealth);

loadSettings();
checkHealth();
