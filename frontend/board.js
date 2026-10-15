// Nurse board: renders the live queue the server pushes over a WebSocket.
// Every critical tops the queue; only an interrupting one plays a sound (alarm principle).

const CATEGORY = {
  breathing: "Solunum", pain: "Ağrı", neuro: "Bilinç", other_symptom: "Diğer belirti",
  fall_injury: "Düşme / yaralanma", physical_help: "Fiziksel yardım",
  comfort_supplies: "Konfor / malzeme", medication: "İlaç", equipment: "Cihaz",
  information: "Bilgi", emotional: "Duygusal", unclear: "Belirsiz",
};
const URGENCY = { critical: "Kritik", soon: "Kısa sürede", routine: "Rutin" };
const RANK = { critical: 2, soon: 1, routine: 0 };

let cards = new Map();
let now = new Date();
let audio = null;

const $ = (id) => document.getElementById(id);

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") node.className = v;
    else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
    else node.setAttribute(k, v);
  }
  for (const child of children) node.append(child);
  return node;
}

function waitText(card) {
  const minutes = Math.max(0, Math.floor((now - new Date(card.created_at)) / 60000));
  return minutes < 1 ? "az önce" : `${minutes} dk`;
}

function reasonText(reasons) {
  // "model: pain/soon (0.91)" -> "model %91" (Turkish puts the percent sign first)
  return reasons.map((r) => r.replace(/^model: \S+ \(([\d.]+)\)$/, (_, p) => `model %${Math.round(p * 100)}`)
                           .replace(/^rule: /, "kural: ")
                           .replace(/^button: /, "buton: ")
                           .replace(/^context: cannot walk alone$/, "tek başına yürüyemiyor"))
                .join(" · ");
}

function render() {
  const queue = [...cards.values()].sort((a, b) =>
    RANK[b.urgency] - RANK[a.urgency] || a.created_at.localeCompare(b.created_at));
  const box = $("queue");
  box.replaceChildren(...queue.map(renderCard));
  $("empty").style.display = queue.length ? "none" : "block";
  for (const u of ["critical", "soon", "routine"]) {
    const n = queue.filter((c) => c.urgency === u).length;
    $(`n-${u}`).textContent = `${n} ${URGENCY[u].toLowerCase()}`;
  }
  $("clock").textContent = now.toLocaleTimeString("tr-TR", { hour: "2-digit", minute: "2-digit" });
}

function renderCard(card) {
  const badges = el("div", { class: "badges" },
    el("span", { class: `badge ${card.urgency}` }, URGENCY[card.urgency]),
    el("span", { class: "badge" }, CATEGORY[card.category] || card.category));
  if (card.uncertain) badges.append(el("span", { class: "badge unsure" }, "Emin değil — kontrol edin"));
  if (card.speaker === "companion") badges.append(el("span", { class: "badge" }, "Refakatçi"));

  const status = card.status === "claimed" ? ` · ${card.claimed_by} ilgileniyor` : "";
  const actions = el("div", { class: "actions" });
  if (card.status === "open") actions.append(el("button", { onclick: () => act(card, "claim") }, "Üstlen"));
  actions.append(el("button", { class: "primary", onclick: () => complete(card) }, "Tamamla"));

  return el("div", { class: `card ${card.urgency} ${card.status}${card.alarm ? " alarm" : ""}` },
    el("div", { class: "where" }, `Oda ${card.room}`, el("small", {}, `Yatak ${card.bed} · ${card.patient}`)),
    el("div", { class: "what" }, badges,
      el("div", { class: "text" }, card.text),
      el("div", { class: "meta" }, el("span", { class: "wait" }, waitText(card)), status, " · ", reasonText(card.reasons))),
    actions);
}

function nurseId() {
  return Number($("nurse").value);
}

async function post(url, body) {
  const res = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" },
                                 body: JSON.stringify(body) });
  if (!res.ok) alert((await res.json()).detail || res.statusText);
  return res.ok;
}

function act(card, action) {
  return post(`/api/requests/${card.id}/${action}`, { nurse_id: nurseId() });
}

async function complete(card) {
  let reallyCritical = null;
  if (card.urgency === "critical") {
    const dialog = $("confirm");
    dialog.showModal();
    reallyCritical = await new Promise((resolve) => {
      dialog.querySelectorAll("button").forEach((b) => (b.onclick = () => { dialog.close(); resolve(b.value === "yes"); }));
    });
  }
  return post(`/api/requests/${card.id}/complete`, { nurse_id: nurseId(), really_critical: reallyCritical });
}

function beep() {
  try {
    audio = audio || new AudioContext();
    for (const [t, f] of [[0, 880], [0.25, 880], [0.5, 1175]]) {
      const osc = audio.createOscillator();
      const gain = audio.createGain();
      osc.frequency.value = f;
      gain.gain.setValueAtTime(0.2, audio.currentTime + t);
      gain.gain.exponentialRampToValueAtTime(0.001, audio.currentTime + t + 0.2);
      osc.connect(gain).connect(audio.destination);
      osc.start(audio.currentTime + t);
      osc.stop(audio.currentTime + t + 0.2);
    }
  } catch (e) { /* no sound before the page has been clicked */ }
}

function onMessage(msg) {
  if (msg.now) now = new Date(msg.now);
  if (msg.type === "snapshot") {
    cards = new Map(msg.cards.map((c) => [c.id, c]));
  } else if (msg.type === "card") {
    const card = msg.card;
    const isNew = !cards.has(card.id);
    if (card.status === "done") cards.delete(card.id);
    else cards.set(card.id, { ...card, alarm: isNew && card.interrupt });
    if (isNew && card.interrupt) beep();
  }
  render();
}

function connect() {
  const ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`);
  ws.onmessage = (e) => onMessage(JSON.parse(e.data));
  ws.onclose = () => setTimeout(connect, 1000);
}

async function init() {
  const state = await (await fetch("/api/state")).json();
  $("nurse").replaceChildren(...state.nurses.map((n) =>
    el("option", { value: n.id }, n.role === "head_nurse" ? `${n.name} (sorumlu)` : n.name)));
  $("replay").disabled = state.replaying;
  $("replay").onclick = async () => {
    if (await post("/api/replay", { speed: Number($("speed").value) })) $("replay").disabled = true;
  };
  connect();
}

init();
