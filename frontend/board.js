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

// Protocol facts in Turkish, for the reason line ("protokol: kusma + koyu kusmuk").
const FACT = {
  breath: "nefes", breath_stopped: "nefes durması", choke: "boğulma", oxygen: "oksijen", chest: "göğüs",
  pain: "ağrı", painkiller: "ağrı kesici", head: "baş", unconscious: "bilinç kaybı", stroke: "inme belirtisi",
  confusion: "bilinç bulanıklığı", dizzy: "baş dönmesi", fever: "ateş/titreme", vomit: "kusma", blood: "kan",
  dark_vomit: "koyu kusmuk", heavy_bleeding: "ağır kanama", vital_dropped: "değer düşmesi", fall: "düşme",
  wound: "yara/pansuman", self_harm: "kendine zarar", hopeless: "umutsuzluk", anxiety: "korku/panik",
  sleep: "uykusuzluk", company: "yalnızlık", toilet: "tuvalet", bedpan: "sürgü", soiled: "kirlenme",
  move_body: "hareket yardımı", hygiene: "temizlik", water: "su", food: "yemek", temperature: "ısı",
  bed: "yatak", environment: "oda ortamı", supplies: "malzeme", medication: "ilaç", missed: "atlanan doz",
  after_drug: "ilaç sonrası", swelling: "şişlik", rash: "kaşıntı/döküntü", device: "cihaz",
  device_question: "cihaz sorusu", iv_site: "damar yolu sorunu", info: "bilgi", cough: "öksürük",
  nose: "burun", constipation: "kabızlık", diarrhea: "ishal", food_question: "yemek sorusu",
  vague: "belirsiz şikâyet", inability: "yapamama", passed: "geçti", severe: "şiddetli", sudden: "ani",
  worsening: "kötüleşme", alarm_sign: "alarm bulgusu", mild: "hafif", urgency_words: "aciliyet ifadesi",
};

function protocolText(reason) {
  // "protocol: line 8: vomit, (blood / dark_vomit) [dark_vomit, vomit]" -> "protokol: kusma + koyu kusmuk"
  const facts = (reason.match(/\[([^\]]*)\]$/) || [, ""])[1].split(", ").filter((f) => f && f !== "no facts");
  const shown = facts.filter((f) => !["urgency_words", "info"].includes(f)).map((f) => FACT[f] || f);
  return shown.length ? `protokol: ${shown.join(" + ")}` : "protokol: tanınmadı";
}

function reasonText(reasons) {
  // "model: pain/soon (0.91)" -> "model %91" (Turkish puts the percent sign first)
  return reasons.map((r) => r.startsWith("protocol: ") ? protocolText(r) : r)
                .map((r) => r.replace(/^model: \S+ \(([\d.]+)\)$/, (_, p) => `model %${Math.round(p * 100)}`)
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
    el("span", { class: `badge ${card.urgency}` }, card.tentative ? "Olası kritik" : URGENCY[card.urgency]),
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

let replay = "idle";  // idle | running | paused | finished

function showReplay(state) {
  replay = state;
  const button = $("replay");
  button.textContent = { idle: "Vardiyayı oynat", running: "Duraklat", paused: "Devam", finished: "Vardiya bitti" }[state];
  button.disabled = state === "finished";
  $("status").textContent = state === "paused" ? "duraklatıldı" : "";
}

async function onReplayButton() {
  if (replay === "idle") {
    await post("/api/replay", { speed: Number($("speed").value) });
  } else if (replay === "running") {
    await post("/api/replay/control", { action: "pause" });
  } else if (replay === "paused") {
    await post("/api/replay/control", { action: "resume" });
  }
}

function onMessage(msg) {
  if (msg.now) now = new Date(msg.now);
  if (msg.replay) showReplay(msg.replay);
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
  showReplay(state.replay);
  $("replay").onclick = onReplayButton;
  // A new speed applies at once while the shift is playing.
  $("speed").onchange = () => {
    if (replay === "running" || replay === "paused") {
      post("/api/replay/control", { action: "speed", speed: Number($("speed").value) });
    }
  };
  connect();
}

init();
