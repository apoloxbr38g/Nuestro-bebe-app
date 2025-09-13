// ===============================
//  APP.JS A PRUEBA DE BALAS 😈
// ===============================

// Usa el mismo origen del backend que sirve /app/
const API = window.location.origin;

// ---- Alias para soportar ambos sets de IDs (viejos y nuevos) ----
const IDS = {
  league: ['sel-league','league'],
  home:   ['sel-home','home'],
  away:   ['sel-away','away'],
};
const byId = (list) => list.map(id => document.getElementById(id)).find(Boolean);

// ---------- Utilidades ----------
const log  = (...a) => console.log("[APP]", ...a);
const warn = (...a) => console.warn("[APP]", ...a);
const err  = (...a) => console.error("[APP]", ...a);
const $    = (id) => document.getElementById(id);

function opt(v){ const o=document.createElement("option"); o.value=v; o.textContent=v; return o; }
function asPct(x){ return Math.max(0, Math.min(100, Math.round(Number(x)*100))); }

// Fetch helpers SIN AbortController (evita “signal is aborted”)
async function getJSON(url){
  log("GET", url);
  try{
    const r = await fetch(url, { cache: "no-store" });
    if(!r.ok){
      const body = await r.text().catch(()=> "");
      throw new Error(`HTTP ${r.status} @ ${url}\n${body}`);
    }
    return await r.json();
  }catch(e){
    err("getJSON ERROR:", e);
    return Promise.reject(e);
  }
}

async function postJSON(url, body){
  log("POST", url, body);
  try{
    const r = await fetch(url, {
      method: "POST",
      headers: {"Content-Type":"application/json"},
      body: JSON.stringify(body || {})
    });
    if(!r.ok){
      const bodyTxt = await r.text().catch(()=> "");
      throw new Error(`HTTP ${r.status} @ ${url}\n${bodyTxt}`);
    }
    return await r.json();
  }catch(e){
    err("postJSON ERROR:", e);
    return Promise.reject(e);
  }
}

// ---------- Estado ----------
let lastPayload = null;

// ====== Ligas con banderitas (valor = código de liga) ======
const LEAGUE_LABELS = {
  SP1: "🇪🇸 LaLiga (ES)",
  E0:  "🇬🇧 Premier (EN)",
  D1:  "🇩🇪 Bundesliga (DE)",
  I1:  "🇮🇹 Serie A (IT)",
  F1:  "🇫🇷 Ligue 1 (FR)",
  N1:  "🇳🇱 Eredivisie (NL)",
  P1:  "🇵🇹 Primeira (PT)",
  SC0: "🇬🇧 Scottish Prem (UK)",
  B1:  "🇧🇪 Jupiler Pro (BE)",
  T1:  "🇹🇷 Superliga (TR)",
  J1:  "🇯🇵 J1 League (JP)",
  G1:  "🇬🇷 Superliga (GR)",
  UKR: "🇺🇦 Premier League (UA)",
};
const LEAGUE_ORDER = ["SP1","E0","D1","I1","F1","N1","P1","SC0","B1","T1","J1","G1","UKR"];

// ========= Pinta el select de ligas con etiquetas bonitas =========
function renderLeagueSelectFromCodes(codes){
  const sel = byId(IDS.league);
  if(!sel) return;

  sel.innerHTML = "";
  const ph = document.createElement("option");
  ph.value = ""; ph.textContent = "— Selecciona liga —";
  ph.disabled = true; ph.selected = true;
  sel.appendChild(ph);

  const ordered = (LEAGUE_ORDER.length ? LEAGUE_ORDER : codes).filter(c => codes.includes(c));
  ordered.forEach(code=>{
    const o = document.createElement("option");
    o.value = code;
    o.textContent = LEAGUE_LABELS[code] || code;
    sel.appendChild(o);
  });

  // al cambiar de liga, cargamos equipos
  sel.onchange = ()=> sel.value && loadTeamsForLeague(sel.value);
}

// ========= Poblar ligas (intenta /leagues/supported, si no /leagues) =========
async function populateLeagues(){
  const sel = byId(IDS.league);
  if(!sel){ warn("No hay select de liga en el DOM"); return []; }

  try{
    // Preferimos el endpoint nuevo
    const sup = await getJSON(`${API}/leagues/supported`);
    const arr = (sup.leagues || []).map(x => x.code);
    if(arr.length){
      renderLeagueSelectFromCodes(arr);
      return arr;
    }
  }catch(_) { /* seguimos al fallback */ }

  // Fallback: /leagues del CSV actual
  try{
    const j = await getJSON(`${API}/leagues`);
    const codes = (j.leagues || []).map(String);
    renderLeagueSelectFromCodes(codes);
    return codes;
  }catch(e){
    console.error("populateLeagues fallback:", e);
    const fallback = Object.keys(LEAGUE_LABELS);
    renderLeagueSelectFromCodes(fallback);
    return fallback;
  }
}

// ========= Cargar equipos por liga (con aliases de IDs) =========
async function loadTeamsForLeague(leagueCode){
  const sH = byId(IDS.home);   // #sel-home o #home
  const sA = byId(IDS.away);   // #sel-away o #away
  if(!sH || !sA){
    warn("No encontré los selects de equipos (home/away)");
    return;
  }
  if(!leagueCode){
    warn("Liga vacía en loadTeamsForLeague");
    sH.innerHTML = ""; sA.innerHTML = "";
    return;
  }

  // placeholder mientras carga
  const makeOpt = (txt, disabled=true, selected=true) => {
    const o = document.createElement("option");
    o.value = ""; o.textContent = txt; o.disabled = !!disabled; o.selected = !!selected;
    return o;
  };
  sH.innerHTML = ""; sA.innerHTML = "";
  sH.appendChild(makeOpt("Cargando…"));
  sA.appendChild(makeOpt("Cargando…"));

  try{
    const data  = await getJSON(`${API}/teams?league=${encodeURIComponent(leagueCode)}`);
    const teams = (data.teams || [])
      .map(t => String(t).trim())
      .filter(Boolean)
      .sort((a,b)=> a.localeCompare(b));

    sH.innerHTML = ""; sA.innerHTML = "";

    if(!teams.length){
      sH.appendChild(makeOpt("Sin equipos", true, true));
      sA.appendChild(makeOpt("Sin equipos", true, true));
      return;
    }

    // primer option «— Selecciona —»
    sH.appendChild(makeOpt("— Selecciona local —", true, true));
    sA.appendChild(makeOpt("— Selecciona visita —", true, true));

    for(const t of teams){
      const o1 = document.createElement("option"); o1.value = t; o1.textContent = t; sH.appendChild(o1);
      const o2 = document.createElement("option"); o2.value = t; o2.textContent = t; sA.appendChild(o2);
    }

    // preselección rápida:
    if(teams.length > 1){
      sH.value = teams[0];
      sA.value = teams[1];
    }
  }catch(e){
    console.error("loadTeamsForLeague:", e);
    sH.innerHTML = ""; sA.innerHTML = "";
    sH.appendChild(makeOpt("Error al cargar", true, true));
    sA.appendChild(makeOpt("Error al cargar", true, true));
    alert("No pude cargar equipos. ¿Backend ok?");
  }
}

// ========= Click en “Cargar ligas” =========
async function handleLoadLeaguesClick(){
  try{
    // elige aquí las ligas que quieres cargar en el backend
    const leagues = ["SP1","E0","D1","I1","F1","N1","P1","SC0","B1","T1","J1","G1","UKR"];
    // 4 temporadas hacia atrás (ajústalo si quieres)
    const start_years = [2024, 2023, 2022, 2021];

    // refresca datos multi-liga en el backend
    const j = await postJSON(`${API}/reload_multi`, { leagues, start_years, build_features: true });
    log("reload_multi OK:", j);

    // repoblamos ligas y precargamos equipos de la primera opción
    await populateLeagues();
    const sel = byId(IDS.league);
    if(sel && sel.options.length > 1){
      sel.selectedIndex = 1;            // primera liga real
      await loadTeamsForLeague(sel.value);
    }
    alert(`Ligas cargadas ✅\nFilas: ${j.rows}\nLigas: ${j.leagues.join(", ")}`);
  }catch(e){
    console.error("handleLoadLeaguesClick:", e);
    alert("No pude recargar datos multi-liga. Revisa la consola.");
  }
}

// --- Helpers Topscorers ---
function normalizeName(p){
  const cands = [
    p.player_name,
    p.player?.name,
    [p.player?.firstname, p.player?.lastname].filter(Boolean).join(" ")
  ].filter(Boolean);

  let name = (cands[0] || "").trim();
  if(!name) name = "—";

  // Elimina duplicados consecutivos
  const parts = name.split(/\s+/);
  const out = [];
  for(const w of parts){
    if(!out.length || out[out.length-1].toLowerCase() !== w.toLowerCase()){
      out.push(w);
    }
  }
  return out.join(" ");
}

function teamName(p){
  return p.team?.name || p.team_name || p.team || "";
}

function playerPhoto(p){
  return p.player?.photo || p.photo ||
    "https://images.unsplash.com/photo-1522778119026-d647f0596c20?q=80&w=240&auto=format&fit=crop";
}

function seasonStartYearForNow(){
  const d = new Date(); const y = d.getFullYear(); const m = d.getMonth() + 1;
  return (m < 8) ? (y - 1) : y; // temporadas europeas inician ~agosto
}

async function loadTopscorers(leagueCode = 'E0', season = seasonStartYearForNow()){
  const box = document.getElementById("topscorers");
  if(!box) return;
  box.textContent = "Cargando goleadores…";

  async function fetchSeason(s){
    const q = new URLSearchParams({ league: leagueCode, season: String(s) });
    return await getJSON(`${API}/topscorers?${q.toString()}`);
  }

  try{
    let data = await fetchSeason(season);
    let list = (data.players || []);

    if(!list.length && season > 2000){
      const prev = season - 1;
      console.log(`[Topscorers] vacío en ${season}, probando ${prev}…`);
      data = await fetchSeason(prev);
      list = (data.players || []);
    }

    if(!list.length){
      box.textContent = "Sin datos de goleadores.";
      return;
    }

    box.innerHTML = list.slice(0,5).map(p => {
      const name  = normalizeName(p);
      const team  = teamName(p);
      const goals = p.goals ?? p.statistics?.[0]?.goals?.total ?? 0;
      const img   = playerPhoto(p);

      return `
        <div class="ts-item">
          <img src="${img}" alt="${name}" />
          <div class="ts-text">
            <div class="ts-name" title="${name}">${name}</div>
            <div class="ts-team" title="${team}">${team}</div>
          </div>
          <div class="ts-goals">
            <b>${goals}</b>
            <span>GOALS</span>
          </div>
        </div>`;
    }).join("");
  }catch(e){
    console.error("Topscorers error:", e);
    box.textContent = "No pude cargar goleadores.";
  }
}

// ---------- Explicaciones ----------
async function openExplain(kind){
  const home = byId(IDS.home)?.value;
  const away = byId(IDS.away)?.value;
  if(!home || !away){ alert("Elige dos equipos, bombón 😘"); return; }

  try{
    const data = await getJSON(`${API}/explain?home=${encodeURIComponent(home)}&away=${encodeURIComponent(away)}`);
    const reasons = (data.reasons||{})[kind] || [];
    alert(`${kind.toUpperCase()} — razones:\n- ${reasons.join("\n- ")}`);
  }catch(e){
    err("Fallo /explain:", e);
    alert("No pude obtener explicación.");
  }
}

// ---------- Render (barras y pildoritas) ----------
function renderVbars1x2(ph, pd, pa){
  const vh = document.getElementById('vbar-home');
  const vd = document.getElementById('vbar-draw');
  const va = document.getElementById('vbar-away');
  const lh = document.getElementById('vlab-home');
  const ld = document.getElementById('vlab-draw');
  const la = document.getElementById('vlab-away');
  if(vh) vh.style.height = Math.max(6, ph) + '%';
  if(vd) vd.style.height = Math.max(6, pd) + '%';
  if(va) va.style.height = Math.max(6, pa) + '%';
  if(lh) lh.textContent = ph + '%';
  if(ld) ld.textContent = pd + '%';
  if(la) la.textContent = pa + '%';
}

function renderHbar(idFill, idLab, pct){
  const f = document.getElementById(idFill);
  const l = document.getElementById(idLab);
  const p = asPct(pct);
  if(f) f.style.width = p + '%';
  if(l) l.textContent = p + '%';
}

// ===== Helpers súper robustos para encontrar métricas en el payload =====
function numOrNull(v){
  const n = Number(v);
  return Number.isFinite(n) ? n : null;
}
function pickFirst(obj, regexes){
  // Busca la primera clave cuyo nombre haga match con cualquiera de los regex
  // Devuelve número o null
  const keys = Object.keys(obj || {});
  for(const rx of regexes){
    const k = keys.find(k => rx.test(k));
    if(k != null){
      const n = numOrNull(obj[k]);
      if(n != null) return n;
    }
  }
  return null;
}

function renderSystemStats(data){
  const setTxt = (id, val) => { const el=document.getElementById(id); if(el) el.textContent = val; };

  // ===== xG =====
  const xgH = numOrNull(data.exp_goals_home) ?? 0;
  const xgA = numOrNull(data.exp_goals_away) ?? 0;
  const xgT = numOrNull(data.exp_goals_total) ?? (xgH + xgA);
  setTxt('stat-xg-h', xgH.toFixed(3));
  setTxt('stat-xg-a', xgA.toFixed(3));
  setTxt('stat-xg-t', xgT.toFixed(3));

  // ===== OU =====
  const pOver = (data.ou_over25 ?? 0);
  const pUnder= (data.ou_under25 ?? (1 - (data.ou_over25 ?? 0)));
  renderHbar('bar-ou-over', 'lab-ou-over', pOver);
  renderHbar('bar-ou-under','lab-ou-under',pUnder);
  setTxt('stat-ou-o', `${asPct(pOver)}%`);
  setTxt('stat-ou-u', `${asPct(pUnder)}%`);

  // ===== BTTS =====
  const pYes = (data.btts_yes ?? 0);
  const pNo  = (data.btts_no  ?? (1 - (data.btts_yes ?? 0)));
  renderHbar('bar-btts-yes', 'lab-btts-yes', pYes);
  renderHbar('bar-btts-no',  'lab-btts-no',  pNo);
  setTxt('stat-btts-y', `${asPct(pYes)}%`);
  setTxt('stat-btts-n', `${asPct(pNo)}%`);

  // ===== CÓRNERS =====
  const cornersH = pickFirst(data, [
    /^(exp_?)?(corners|corner|avg_?corners).*home$/i,
    /^(exp_?)?(corners|corner|avg_?corners).*_h$/i,
    /^(exp_?)?corners?_?local$/i
  ]);
  const cornersA = pickFirst(data, [
    /^(exp_?)?(corners|corner|avg_?corners).*away$/i,
    /^(exp_?)?(corners|corner|avg_?corners).*_a$/i,
    /^(exp_?)?corners?_?(visit|away|visita)$/i
  ]);
  if (cornersH != null) setTxt('stat-c-h', cornersH.toFixed(1));
  if (cornersA != null) setTxt('stat-c-a', cornersA.toFixed(1));

  // ===== AMARILLAS =====
  const yellH = pickFirst(data, [
    /^(exp_?)?(yellow|yellows|cards?_?yellow|amarillas?).*home$/i,
    /^(exp_?)?(yellow|yellows|cards?_?yellow|amarillas?).*_h$/i,
    /^(exp_?)?amarillas?_?local$/i
  ]);
  const yellA = pickFirst(data, [
    /^(exp_?)?(yellow|yellows|cards?_?yellow|amarillas?).*away$/i,
    /^(exp_?)?(yellow|yellows|cards?_?yellow|amarillas?).*_a$/i,
    /^(exp_?)?amarillas?_?(visit|away|visita)$/i
  ]);
  if (yellH != null) setTxt('stat-y-h', yellH.toFixed(1));
  if (yellA != null) setTxt('stat-y-a', yellA.toFixed(1));

  // ===== ROJAS =====
  const redH = pickFirst(data, [
    /^(exp_?)?(red|reds|cards?_?red|rojas?).*home$/i,
    /^(exp_?)?(red|reds|cards?_?red|rojas?).*_h$/i,
    /^(exp_?)?rojas?_?local$/i
  ]);
  const redA = pickFirst(data, [
    /^(exp_?)?(red|reds|cards?_?red|rojas?).*away$/i,
    /^(exp_?)?(red|reds|cards?_?red|rojas?).*_a$/i,
    /^(exp_?)?rojas?_?(visit|away|visita)$/i
  ]);
  if (redH != null) setTxt('stat-r-h', redH.toFixed(2));
  if (redA != null) setTxt('stat-r-a', redA.toFixed(2));
}

// --- Scorelines (defensa y compatibilidad) ---
if (typeof window.renderScorelines !== 'function') {
  window.renderScorelines = function renderScorelines(listId, arr) {
    const list = document.getElementById(listId);
    if (!list) return;
    list.innerHTML = "";
    (arr || []).forEach(s => {
      const p = Number(s.p ?? s.prob ?? s.probability ?? 0);
      const scoreTxt = s.score || (s.h != null && s.a != null ? `${s.h}-${s.a}` : "—");
      const li = document.createElement("li");
      li.textContent = `${scoreTxt} — ${(p * 100).toFixed(2)}%`;
      list.appendChild(li);
    });
  };
}

// ---------- Predicción (barras verticales + pildoritas) ----------
async function predict(){
  log("predict() llamado");

  const home = byId(IDS.home)?.value;
  const away = byId(IDS.away)?.value;

  if(!home || !away){ alert("Elige dos equipos, bombón 😘"); return; }
  if(home === away){ alert("Elige equipos distintos, bombón 😘"); return; }

  const url = `${API}/predict?home=${encodeURIComponent(home)}&away=${encodeURIComponent(away)}`;
  log("GET", url);

  try{
    const data = await getJSON(url);
    lastPayload = data;

    // 🔎 Para depurar las píldoras: mira las claves reales que manda el backend
    console.log("predict payload:", data);

    // 1X2 → barras verticales
    const ph = asPct(data.p_home),
          pd = asPct(data.p_draw),
          pa = asPct(data.p_away);
    renderVbars1x2(ph, pd, pa);

    // xG (texto)
    if ($("xg-home")) $("xg-home").textContent = Number(data.exp_goals_home||0).toFixed(3);
    if ($("xg-away")) $("xg-away").textContent = Number(data.exp_goals_away||0).toFixed(3);
    const xgt = (typeof data.exp_goals_total === 'number')
      ? data.exp_goals_total
      : (Number(data.exp_goals_home||0) + Number(data.exp_goals_away||0));
    if ($("xg-total")) $("xg-total").textContent = Number(xgt).toFixed(3);

    // OU + BTTS + (córners/amarillas/rojas) → barras/píldoras
    renderSystemStats(data);

    // Marcadores probables (si vienen)
    if (typeof renderScorelines === 'function') {
      renderScorelines("scorelines", data.top_scorelines || []);
    }

    // Mostrar el bloque de resultados si estaba oculto
    $("results")?.classList.remove("hidden");

  }catch(e){
    err("predict() fallo:", e);
    alert("Fallo /predict:\n" + (e?.message || e));
  }
}

// ---------- Value Finder ----------
function impliedProb(odds){ return odds>0 ? 1/odds : NaN; }
function edge(ourP, odds){ const imp = impliedProb(odds); return isNaN(imp) ? NaN : (ourP - imp); }

function showValue(listId, items){
  const ul = $(listId);
  if(!ul) return;
  ul.innerHTML = "";
  items.forEach(x=>{
    const li = document.createElement("li");
    const pct = (x.edge*100).toFixed(2);
    li.textContent = `${x.label}: edge ${pct}% (nuestra p=${(x.our*100).toFixed(1)}%, implícita=${(x.imp*100).toFixed(1)}%)`;
    li.className = (x.edge>0 ? 'good' : 'bad');
    ul.appendChild(li);
  });
}

function calcValue1x2(){
  if(!lastPayload){ alert("Primero haz una predicción."); return; }
  const {p_home,p_draw,p_away} = lastPayload;
  const oh = parseFloat($("odds-h")?.value || "");
  const od = parseFloat($("odds-d")?.value || "");
  const oa = parseFloat($("odds-a")?.value || "");
  const items = [];
  if(oh) items.push({label:'Local',    our:p_home, imp:impliedProb(oh), edge:edge(p_home,oh)});
  if(od) items.push({label:'Empate',   our:p_draw, imp:impliedProb(od), edge:edge(p_draw,od)});
  if(oa) items.push({label:'Visitante',our:p_away, imp:impliedProb(oa), edge:edge(p_away,oa)});
  showValue("value-1x2", items);
}

function calcValueOU(){
  if(!lastPayload){ alert("Primero haz una predicción."); return; }
  const {ou_over25,ou_under25} = lastPayload;
  const oov = parseFloat($("odds-over")?.value || "");
  const oud = parseFloat($("odds-under")?.value || "");
  const items = [];
  if(oov) items.push({label:'Over 2.5',  our:ou_over25, imp:impliedProb(oov), edge:edge(ou_over25,oov)});
  if(oud) items.push({label:'Under 2.5', our:ou_under25, imp:impliedProb(oud), edge:edge(ou_under25,oud)});
  showValue("value-ou", items);
}

function calcValueBTTS(){
  if(!lastPayload){ alert("Primero haz una predicción."); return; }
  const {btts_yes,btts_no} = lastPayload;
  const oy = parseFloat($("odds-btts-yes")?.value || "");
  const on = parseFloat($("odds-btts-no")?.value  || "");
  const items = [];
  if(oy) items.push({label:'BTTS Sí', our:btts_yes, imp:impliedProb(oy), edge:edge(btts_yes,oy)});
  if(on) items.push({label:'BTTS No', our:btts_no,  imp:impliedProb(on), edge:edge(btts_no,on)});
  showValue("value-btts", items);
}

// ---------- Recarga multi-liga manual (extra) ----------
async function reloadMulti(){
  try{
    const leagues = ['SP1','E0','I1'];
    const start_years = [2024, 2023, 2022, 2021];
    const j = await postJSON(`${API}/reload_multi`, { leagues, start_years });
    alert(`Datos recargados: ${j.rows} filas\nLigas: ${j.leagues.join(', ')}\nAños: ${j.years.join(', ')}`);
    await init();
  }catch(e){
    err("Fallo reload_multi:", e);
    alert("No pude recargar datos multi-liga.");
  }
}

// ---------- Router de vistas ----------
const VIEWS = {
  home:        "view-home",
  prediccion:  "view-prediccion",
  estado:      "view-estado",
  config:      "view-config",
};

function showView(page){
  // Oculta todas
  Object.values(VIEWS).forEach(id => { const el = $(id); if(el) el.classList.add("hidden"); });

  // Muestra la pedida
  const vid = VIEWS[page] || VIEWS.home;
  const view = $(vid);
  if(view) view.classList.remove("hidden");

  // 👉 dispara el chequeo cuando entras a “Estado del Sistema”
  if (page === "estado") {
    loadSystemStatus();
  }

  // Marca el botón activo
  document.querySelectorAll(".nav-btn").forEach(b => b.classList.remove("active"));
  const btn = document.querySelector(`.nav-btn[data-page="${page}"]`);
  if(btn) btn.classList.add("active");

  // Ajusta hash
  if(location.hash.replace(/^#\/?/, "") !== page){
    history.replaceState(null, "", `#/${page}`);
  }
}

function getPageFromHash(){
  const raw = location.hash.replace(/^#\/?/, "");
  return (raw && VIEWS[raw]) ? raw : "home";
}

// Delegación de clicks en la sidebar
document.addEventListener("click", (e)=>{
  const btn = e.target.closest(".nav-btn[data-page]");
  if(!btn) return;
  e.preventDefault();
  const page = btn.getAttribute("data-page");
  showView(page);
});

// React a cambios del hash
window.addEventListener("hashchange", ()=> showView(getPageFromHash()) );

// ---------- Enganches ----------
function bindPredictButton(){
  const b = $("btn") || $("btn-predict");
  if(!b){ warn("No encontré botón #btn/#btn-predict. Reintentando…"); setTimeout(bindPredictButton, 300); return; }
  b.addEventListener("click", (e)=>{ e.preventDefault(); predict(); });
}

function bindOthers(){
  $("reload")?.addEventListener("click", reloadMulti);
  $("btn-value-1x2")?.addEventListener("click", calcValue1x2);
  $("btn-value-ou")?.addEventListener("click",  calcValueOU);
  $("btn-value-btts")?.addEventListener("click", calcValueBTTS);
  $("exp-1x2")?.addEventListener("click", ()=>openExplain("1x2"));
  $("exp-ou")?.addEventListener("click",  ()=>openExplain("ou25"));
  $("exp-btts")?.addEventListener("click",()=>openExplain("btts"));

  // 👉 aquí añadimos el botón de refrescar estado
  $("btn-refresh-status")?.addEventListener("click", (e)=>{
    e.preventDefault();
    loadSystemStatus();
  });
}

// Enganches específicos de la vista Predicción
function bindPrediccionUI(){
  // Botón “Cargar ligas”
  document.getElementById('btn-load-leagues')
    ?.addEventListener('click', (e)=>{ e.preventDefault(); handleLoadLeaguesClick(); });

  // Select de liga: al cambiar, cargar equipos
  const sel = byId(IDS.league);
  sel?.addEventListener('change', ()=> sel.value && loadTeamsForLeague(sel.value));

  // Si entramos y está vacío el select, lo poblamos
  if(sel && !sel.options.length){
    populateLeagues().then(()=>{
      if(sel.options.length > 1){
        sel.selectedIndex = 1;
        loadTeamsForLeague(sel.value);
      }
    });
  }
}

// ---------- Últimos 10 partidos ----------
async function loadResults(){
  const tbody = document.getElementById("rows-results");
  if(!tbody) return;
  tbody.innerHTML = '<tr><td colspan="5" style="opacity:.6">Cargando…</td></tr>';

  try{
    const j = await getJSON(`${API}/recent_live?days=8&limit=10`);
    const rows = (j.matches || []).filter(m => m.FTHG!=null && m.FTAG!=null);
    if(!rows.length){
      tbody.innerHTML = '<tr><td colspan="5" style="opacity:.6">No hay resultados.</td></tr>';
      return;
    }
    tbody.innerHTML = rows.map(m=>{
      const score = `${m.FTHG} – ${m.FTAG}`;
      let clsH='draw', clsA='draw';
      if(m.FTHG>m.FTAG){ clsH='win'; clsA='loss'; }
      else if(m.FTHG<m.FTAG){ clsH='loss'; clsA='win'; }
      return `
        <tr>
          <td>${m.Date}</td>
          <td>${m.League}</td>
          <td class="${clsH}">${m.HomeTeam}</td>
          <td class="score">${score}</td>
          <td class="${clsA}">${m.AwayTeam}</td>
        </tr>`;
    }).join('');
  }catch(e){
    console.error("loadResults:", e);
    tbody.innerHTML = '<tr><td colspan="5" style="color:#fca5a5">Error al cargar.</td></tr>';
  }
}

// ---------- Próximos partidos ----------
async function loadFixtures(){
  const tbody = document.getElementById("rows-fixtures");
  if(!tbody) return;
  tbody.innerHTML = '<tr><td colspan="5" style="opacity:.6">Cargando…</td></tr>';

  try{
    const j = await getJSON(`${API}/fixtures/global_next5?days=14`);
    const rows = j.fixtures || [];
    if(!rows.length){
      tbody.innerHTML = '<tr><td colspan="5" style="opacity:.6">No hay próximos partidos.</td></tr>';
      return;
    }
    tbody.innerHTML = rows.map(m=>`
      <tr>
        <td>${m.datetime.split(" ")[0]}</td>
        <td>${m.league}</td>
        <td>${m.home}</td>
        <td>${m.away}</td>
        <td>${m.status || "Pendiente"}</td>
      </tr>`).join('');
  }catch(e){
    console.error("loadFixtures:", e);
    tbody.innerHTML = '<tr><td colspan="5" style="color:#fca5a5">Error al cargar.</td></tr>';
  }
}

async function loadSystemStatus(){
  const box = document.getElementById("system-status");
  if(!box) return;
  box.textContent = "Verificando estado del sistema…";

  // Helpers de UI
  const okBadge = (txt="OK") => `<span style="padding:2px 8px;border-radius:999px;background:#0c2a18;color:#34d399;border:1px solid #134e4a">${txt}</span>`;
  const badBadge= (txt="ERROR") => `<span style="padding:2px 8px;border-radius:999px;background:#2a1212;color:#f87171;border:1px solid #7f1d1d">${txt}</span>`;

  try{
    // /health — rápido
    let health = null;
    try { health = await getJSON(`${API}/health`); } catch { /* seguimos */ }
    const healthy = !!(health && (health.status === "ok" || health.ok));

    // /bot_status — modelos, csv, etc.
    let bot = null;
    try { bot = await getJSON(`${API}/bot_status`); } catch { /* seguimos */ }

    // /metrics — filas, ligas, equipos
    let m = null;
    try { m = await getJSON(`${API}/metrics`); } catch { /* seguimos */ }

    let html = `
      <h3 style="margin:0 0 8px; color:${healthy ? '#34d399' : '#f87171'}">
        Backend ${healthy ? 'operativo' : 'con problemas'} ${healthy ? '✅' : '❌'}
      </h3>
      <ul style="margin:0; padding-left:18px; line-height:1.6">
        <li><b>Health:</b> ${healthy ? okBadge('OK') : badBadge('NO')}</li>
    `;

    if(bot){
      html += `
        <li><b>Datos:</b> ${bot.data_rows ?? 0} filas ${bot.features_ready ? okBadge('features listas') : badBadge('features no listas')}</li>
        <li><b>Modelos:</b>
          Poisson ${bot.models?.poisson ? okBadge() : badBadge()}
          · 1X2 ${bot.models?.ml_1x2 ? okBadge() : badBadge()}
          · O/U ${bot.models?.ml_ou25 ? okBadge() : badBadge()}
          · BTTS ${bot.models?.ml_btts ? okBadge() : badBadge()}
        </li>
        <li><b>CSV actual:</b> ${bot.current_csv ?? '—'}</li>
      `;
    }

    if(m && m.status !== 'empty'){
      html += `
        <li><b>Partidos:</b> ${m.matches} · <b>Equipos:</b> ${m.n_teams}</li>
        <li><b>Algunas ligas:</b> ${(m.leagues || []).slice(0,8).join(', ') || '—'}</li>
      `;
    }

    html += `</ul>`;

    box.innerHTML = html;
  }catch(e){
    console.error("loadSystemStatus:", e);
    box.innerHTML = `<h3 style="color:#f87171">Backend no responde ❌</h3>
      <p>Revisa si <code>/health</code> está disponible.</p>`;
  }
}

// ---------- Boot ----------
async function init(){
  log("Init UI…");

  // Conectividad
  try{
    const ping = await getJSON(`${API}/health`);
    log("/health OK:", ping);
  }catch(e){
    err("/health FALLÓ:", e);
    alert("No pude hablar con el backend (/health). Abre http://127.0.0.1:8000/health para verificar.");
  }

  // Vista según hash
  showView(getPageFromHash());

  // Si hay select de liga, poblar en frío
  await populateLeagues();
  const sel = byId(IDS.league);
  if(sel){
    let code = sel.value;
    if(!code && sel.options.length > 1){ // primer real tras placeholder
      sel.selectedIndex = 1;
      code = sel.value;
    }
    if(code) await loadTeamsForLeague(code);
  }
}

document.addEventListener("DOMContentLoaded", async () => {
  try{
    bindPredictButton();
    bindOthers();
    bindPrediccionUI();
    await init();
    await loadResults();
    await loadFixtures();
    await loadTopscorers('E0'); // Premier por defecto (cámbialo si quieres)
    log("App lista 💋");
  }catch(e){
    err("Error inicializando app:", e);
    alert("Se rompió el JS al iniciar. Revisa la consola (F12 > Console).");
  }
});
