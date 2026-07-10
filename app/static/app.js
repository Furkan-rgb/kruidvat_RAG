"use strict";

const form = document.querySelector("#ask-form");
const question = document.querySelector("#question");
const mode = document.querySelector("#mode");
const topK = document.querySelector("#top-k");
const submitButton = document.querySelector("#submit-button");
const loading = document.querySelector("#loading");
const loadingMessage = document.querySelector("#loading-message");
const errorBox = document.querySelector("#error");
const results = document.querySelector("#results");
const answerText = document.querySelector("#answer-text");
const answerMode = document.querySelector("#answer-mode");
const resultMeta = document.querySelector("#result-meta");
const productList = document.querySelector("#product-list");
const healthStatus = document.querySelector("#health-status");

let submitting = false;

const remediationByStatus = {
  database_missing: "Run python scraper.py to create the catalogue.",
  products_table_missing: "Run python scraper.py to initialize the products table.",
  vector_index_missing: "Run python embed.py to build the vector index.",
  embedding_index_mismatch: "Run python embed.py --reset with the configured embedding provider.",
  sqlite_vec_unavailable: "Reinstall sqlite-vec and check SQLite extension support.",
  database_unavailable: "Check DB_PATH and database file permissions."
};

function setHealth(label, className, title = "") {
  healthStatus.className = `status ${className}`;
  healthStatus.replaceChildren();
  const dot = document.createElement("span");
  dot.className = "status-dot";
  dot.setAttribute("aria-hidden", "true");
  const text = document.createElement("span");
  text.textContent = label;
  healthStatus.append(dot, text);
  healthStatus.title = title;
}

async function checkHealth() {
  try {
    const response = await fetch("/api/health");
    if (!response.ok) throw new Error("Health request failed");
    const health = await response.json();
    if (health.status === "ready") {
      setHealth("Ready", "status-ready", `${health.embedded_product_count ?? 0} products embedded`);
      return;
    }
    const labels = {
      database_missing: "Database missing",
      products_table_missing: "Products missing",
      vector_index_missing: "Vector index missing",
      embedding_index_mismatch: "Embedding index mismatch",
      sqlite_vec_unavailable: "Setup incomplete",
      database_unavailable: "Setup incomplete"
    };
    setHealth(labels[health.status] || "Setup incomplete", "status-warning", remediationByStatus[health.status] || "Check the local setup.");
  } catch (_error) {
    setHealth("Server unavailable", "status-error", "Start the FastAPI application and reload.");
  }
}

function setBusy(busy) {
  submitting = busy;
  loading.hidden = !busy;
  for (const control of form.elements) control.disabled = busy;
  submitButton.textContent = busy ? "Working…" : "Ask advisor";
  if (busy) loadingMessage.textContent = "Preparing your request...";
}

function showError(message, remediation = "") {
  errorBox.replaceChildren();
  const strong = document.createElement("strong");
  strong.textContent = message;
  errorBox.append(strong);
  if (remediation) {
    const detail = document.createElement("p");
    detail.textContent = remediation;
    errorBox.append(detail);
  }
  errorBox.hidden = false;
}

function ingredientDetails(product) {
  const details = document.createElement("details");
  details.className = "ingredients";
  const summary = document.createElement("summary");
  summary.textContent = `Ingredients (${product.ingredients.length})`;
  const text = document.createElement("p");
  text.textContent = product.ingredients.length ? product.ingredients.join(", ") : "No parsed ingredient list available.";
  details.append(summary, text);
  return details;
}

function catalogueText(value) {
  // Catalogue descriptions contain harmless formatting tags from the OCC API.
  // Parse them in a detached document, then render only their text content.
  return new DOMParser().parseFromString(value, "text/html").body.textContent || "";
}

function appendInlineMarkdown(parent, source) {
  // Deliberately limited inline Markdown: only **bold** is interpreted. All
  // content is appended as text nodes, so model output can never inject HTML.
  const boldPattern = /\*\*(.+?)\*\*/g;
  let cursor = 0;
  let match;
  while ((match = boldPattern.exec(source)) !== null) {
    if (match.index > cursor) parent.append(document.createTextNode(source.slice(cursor, match.index)));
    const strong = document.createElement("strong");
    strong.textContent = match[1];
    parent.append(strong);
    cursor = match.index + match[0].length;
  }
  if (cursor < source.length) parent.append(document.createTextNode(source.slice(cursor)));
}

function renderAnswerMarkdown(source) {
  // This is intentionally not a complete Markdown parser. It covers the small
  // predictable subset emitted by the answer model without accepting raw HTML.
  answerText.replaceChildren();
  let list = null;
  let paragraphLines = [];

  function flushParagraph() {
    if (!paragraphLines.length) return;
    const paragraph = document.createElement("p");
    appendInlineMarkdown(paragraph, paragraphLines.join(" "));
    answerText.append(paragraph);
    paragraphLines = [];
  }

  function endList() {
    list = null;
  }

  for (const rawLine of source.replace(/\r\n?/g, "\n").split("\n")) {
    const line = rawLine.trim();
    if (!line) {
      flushParagraph();
      endList();
      continue;
    }

    const heading = line.match(/^(#{1,4})\s+(.+)$/);
    if (heading) {
      flushParagraph();
      endList();
      const level = Math.min(Math.max(heading[1].length, 3), 4);
      const element = document.createElement(`h${level}`);
      appendInlineMarkdown(element, heading[2]);
      answerText.append(element);
      continue;
    }

    const unordered = line.match(/^[-*+]\s+(.+)$/);
    const ordered = line.match(/^\d+[.)]\s+(.+)$/);
    if (unordered || ordered) {
      flushParagraph();
      const tag = ordered ? "OL" : "UL";
      if (!list || list.tagName !== tag) {
        list = document.createElement(tag.toLowerCase());
        answerText.append(list);
      }
      const item = document.createElement("li");
      appendInlineMarkdown(item, (ordered || unordered)[1]);
      list.append(item);
      continue;
    }

    endList();
    paragraphLines.push(line);
  }
  flushParagraph();
}

function renderProduct(product) {
  const article = document.createElement("article");
  article.className = "product-card";

  const meta = document.createElement("div");
  meta.className = "product-meta";
  const rank = document.createElement("span");
  rank.className = "rank";
  rank.textContent = `#${product.rank}`;
  const evidenceLabel = document.createElement("span");
  evidenceLabel.textContent = "Retrieved catalogue evidence";
  meta.append(rank, evidenceLabel);

  const title = document.createElement("h3");
  title.textContent = product.name || "Unnamed product";
  const distance = document.createElement("p");
  distance.className = "distance";
  distance.textContent = `Vector distance ${Number(product.distance).toFixed(3)}`;
  article.append(meta, title, distance);

  if (product.description) {
    const description = document.createElement("p");
    description.className = "description";
    description.textContent = catalogueText(product.description);
    article.append(description);
  }
  article.append(ingredientDetails(product));

  if (product.url) {
    const link = document.createElement("a");
    link.className = "source-link";
    link.href = product.url;
    link.target = "_blank";
    link.rel = "noopener noreferrer";
    link.textContent = "View catalogue source ↗";
    article.append(link);
  }
  return article;
}

function renderEvidence(data) {
  answerMode.textContent = data.mode === "strict" ? "Strict mode" : "Advisor mode";
  resultMeta.textContent = `${data.products.length} retrieved · answer streaming`;
  productList.replaceChildren(...data.products.map(renderProduct));
  if (!data.products.length) {
    const empty = document.createElement("p");
    empty.className = "empty-state";
    empty.textContent = "No catalogue evidence was retrieved. If products should be available, run python embed.py.";
    productList.append(empty);
  }
  answerText.textContent = "Waiting for the answer model...";
  results.hidden = false;
  results.scrollIntoView({ behavior: "smooth", block: "start" });
}

async function consumeAnswerStream(response) {
  if (!response.body) throw new Error("Streaming responses are not supported by this browser.");
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let streamedAnswer = "";
  let productCount = 0;
  let completed = false;

  async function handleLine(line) {
    if (!line.trim()) return true;
    const event = JSON.parse(line);
    if (event.type === "status") {
      loadingMessage.textContent = event.message;
      return true;
    }
    if (event.type === "evidence") {
      productCount = event.products.length;
      renderEvidence(event);
      return true;
    }
    if (event.type === "token") {
      streamedAnswer += event.text;
      renderAnswerMarkdown(streamedAnswer);
      results.hidden = false;
      return true;
    }
    if (event.type === "done") {
      streamedAnswer = event.answer || streamedAnswer;
      renderAnswerMarkdown(streamedAnswer);
      resultMeta.textContent = `${productCount} retrieved · ${Math.round(event.elapsed_ms)} ms`;
      loadingMessage.textContent = "Answer complete.";
      completed = true;
      return true;
    }
    if (event.type === "error") {
      showError(event.message || "The streamed request failed.", event.remediation || "");
      return false;
    }
    return true;
  }

  while (true) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
    const lines = buffer.split("\n");
    buffer = lines.pop() || "";
    for (const line of lines) {
      if (!await handleLine(line)) {
        await reader.cancel();
        return false;
      }
    }
    if (done) break;
  }
  if (buffer && !await handleLine(buffer)) return false;
  if (!completed) throw new Error("The answer stream ended before completion.");
  return true;
}

async function parseError(response) {
  let body;
  try { body = await response.json(); } catch (_error) { return { message: "The server returned an unreadable response." }; }
  if (Array.isArray(body.detail)) {
    return { message: body.detail.map((item) => item.msg).join(" "), remediation: "Check the question, mode, and top-k values." };
  }
  if (body.detail && typeof body.detail === "object") return body.detail;
  return { message: typeof body.detail === "string" ? body.detail : "The request failed." };
}

async function submitQuestion(event) {
  event.preventDefault();
  if (submitting) return;
  const value = question.value.trim();
  if (!value) {
    showError("Enter a question before asking.");
    question.focus();
    return;
  }
  errorBox.hidden = true;
  results.hidden = true;
  setBusy(true);
  try {
    const response = await fetch("/api/ask/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: value, mode: mode.value, top_k: Number(topK.value) })
    });
    if (!response.ok) {
      const detail = await parseError(response);
      showError(detail.message || "The request failed.", detail.remediation || "");
      return;
    }
    await consumeAnswerStream(response);
  } catch (error) {
    const timedOut = error && error.name === "AbortError";
    const streamFailed = error && String(error.message).toLowerCase().includes("stream");
    showError(
      timedOut ? "The request timed out." : streamFailed ? "The answer stream ended unexpectedly." : "The application server could not be reached.",
      timedOut || streamFailed ? "Check Ollama and try again." : "Make sure Uvicorn is running, then retry."
    );
  } finally {
    setBusy(false);
  }
}

form.addEventListener("submit", submitQuestion);
question.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    form.requestSubmit();
  }
});
document.querySelectorAll(".example").forEach((button) => {
  button.addEventListener("click", () => {
    question.value = button.dataset.question || "";
    question.focus();
  });
});

checkHealth();
