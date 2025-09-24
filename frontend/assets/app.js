/* =========================================================================
   InterfazV1 — app.js  (roles + login + UI completa)
   ========================================================================= */

// ---------- Utilidades ----------
const log  = (...a) => console.log("[APP]", ...a);
const warn = (...a) => console.warn("[APP]", ...a);
const err  = (...a) => console.error("[APP]", ...a);
const $id  = (id) => document.getElementById(id);

function authHeader(){
  const token = localStorage.getItem("sp_token");
  // OJO: "Bearer " con espacio y B mayúscula
  return token ? { Authorization: "Bearer " + token } : {};
}

async function getJSON(url){
  log("GET", url);
  try{
    const headers = { ...authHeader() };
    const r = await fetch(url, { headers, cache: "no-store" });
    if(!r.ok){
      // Si el token está mal/expiró, desloguea suave
      if (r.status === 401) softLogout();
      const body = await r.text().catch(()=> "");
      throw new Error(`HTTP ${r.status} @ ${url}\n${body}`);
    }
    return await r.json();
  }catch(e){ err("getJSON ERROR:", e); throw e; }
}

async function postJSON(url, body){
  log("POST", url, body);
  try{
    const headers = { "Content-Type":"application/json", ...authHeader() };
    const r = await fetch(url, {
      method: "POST",
      headers,
      body: JSON.stringify(body || {})
    });
    if(!r.ok){
      if (r.status === 401) softLogout();
      const bodyTxt = await r.text().catch(()=> "");
      throw new Error(`HTTP ${r.status} @ ${url}\n${bodyTxt}`);
    }
    return await r.json();
  }catch(e){ err("postJSON ERROR:", e); throw e; }
}

function softLogout(){
  // No abras modales, solo limpia y refresca UI a "guest"
  localStorage.removeItem("sp_token");
  localStorage.setItem("sp_role", "guest");
  // Si tienes una función refreshUserUI, llama:
  if (typeof refreshUserUI === "function") refreshUserUI("guest");
}

/* ---------- AUTH / ROLES ---------- */
const ROLES = {
  guest: ["home"],                               // solo Home
  user:  ["home", "prediccion", "estado"],       // sin Config
  admin: ["home", "prediccion", "estado", "config"]
};
function currentRole(){ return localStorage.getItem("sp_role") || "guest"; }
function currentUser(){ return localStorage.getItem("sp_user") || "Usuario"; }
function canAccess(page){ return ROLES[currentRole()]?.includes(page); }

function applyRoleAccess(){
  const role = currentRole();
  document.querySelectorAll(".nav-btn").forEach(btn=>{
    const page = btn.dataset.page;
    if(!ROLES[role].includes(page)) btn.classList.add("hidden");
    else btn.classList.remove("hidden");
  });
  // Si está en una vista no permitida, forzar home
  const page = (location.hash.replace(/^#\/?/, "") || "home");
  if(!ROLES[role].includes(page)){
    showView("home"); history.replaceState(null,"","#/home");
  }
}

function refreshUserUI(roleOverride){
  const role = roleOverride || currentRole();
  const user = currentUser();

  const title = $id("title");
  const subtitle = $id("subtitle");
  if (title)    title.textContent = role === "guest" ? "Bienvenido" : `Bienvenido, ${user}..`;
  if (subtitle) subtitle.textContent = role === "guest"
      ? "Por favor inicia sesión para más funciones."
      : (role === "admin" ? "Tienes acceso administrador." : "Tienes acceso de usuario.");

  const box   = $id("login-box");
  const btnLo = $id("btn-logout");
  const btnIn = $id("login-submit");
  const uIn   = $id("login-user");
  const pIn   = $id("login-pass");
  const errL  = $id("login-error");
  if (errL) errL.textContent = "";

  if(role === "guest"){
    box?.classList.remove("hidden");
    if (btnLo) btnLo.style.display = "none";
    if (btnIn) btnIn.style.display = "";
    if (uIn) uIn.disabled = false;
    if (pIn) pIn.disabled = false;
  }else{
    if (btnLo) btnLo.style.display = "";
    if (btnIn) btnIn.style.display = "none";
    if (uIn) uIn.disabled = true;
    if (pIn) pIn.disabled = true;
  }
  applyRoleAccess();
}

async function doLogin(){
  const u = ($id("login-user")?.value || "").trim();
  const p = ($id("login-pass")?.value || "").trim();
  const errL = $id("login-error");
  if(!u || !p){ if (errL) errL.textContent = "Usuario y contraseña requeridos."; return; }
  try{
    const r = await postJSON("/auth/login", { username: u, password: p });
    localStorage.setItem("sp_token", r.access_token);
    localStorage.setItem("sp_role",  r.role || "user");
    localStorage.setItem("sp_user",  u);
    if ($id("login-pass")) $id("login-pass").value = "";
    refreshUserUI(r.role || "user");
    routeToHash(); // re-evalúa permisos de la vista actual
  }catch(e){
    if (errL) errL.textContent = "Credenciales inválidas.";
  }
}

function doLogout(){
  localStorage.removeItem("sp_token");
  localStorage.removeItem("sp_role");
  localStorage.removeItem("sp_user");
  refreshUserUI("guest");
  showView("home"); history.replaceState(null,"","#/home");
}

/* ---------- Enrutado / vistas ---------- */
function showView(page){
  document.querySelectorAll(".view").forEach(v => v.classList.add("hidden"));
  const el = $id("view-" + page);
  if (el) el.classList.remove("hidden");
  document.querySelectorAll(".nav-btn").forEach(b=>{
    b.classList.toggle("active", b.dataset.page === page);
  });
  // Carga perezosa por vista
  if(page === "home")       loadHome().catch(()=>{});
  if(page === "prediccion") ensureLeagueListsBound();
  if(page === "estado")     bindSystemStatusOnce();
}

function routeToHash(){
  const page = (location.hash.replace(/^#\/?/, "") || "home");
  if(!canAccess(page)){ showView("home"); history.replaceState(null,"","#/home"); return; }
  showView(page);
}

/* ---------- HOME (resultados, próximos, goleadores) ---------- */
async function loadHome(){
  // Últimos 10 resultados
  try{
    const data = await getJSON("/recent?limit=10");
    const tbody = $id("rows-results");
    if (tbody) {
      tbody.innerHTML = "";
      (data.matches || []).forEach(m=>{
        const tr = document.createElement("tr");
        const sc = (m.FTHG==null || m.FTAG==null) ? "—" : `${m.FTHG} – ${m.FTAG}`;
        tr.innerHTML = `
          <td>${m.Date||""}</td>
          <td>${m.League||""}</td>
          <td>${m.HomeTeam||""}</td>
          <td class="score">${sc}</td>
          <td>${m.AwayTeam||""}</td>`;
        tbody.appendChild(tr);
      });
    }
  }catch(e){ warn("recent fail", e); }

  // Próximos 5 partidos
  try{
    const data = await getJSON("/fixtures/global_next5?days=14");
    const tbody = $id("rows-fixtures");
    if (tbody) {
      tbody.innerHTML = "";
      (data.fixtures || []).forEach(m=>{
        const tr = document.createElement("tr");
        tr.innerHTML = `
          <td>${(m.datetime||"").split(" ")[0]}</td>
          <td>${m.league||""}</td>
          <td>${m.home||""}</td>
          <td>${m.away||""}</td>
          <td>${m.status||"Pendiente"}</td>`;
        tbody.appendChild(tr);
      });
    }
  }catch(e){ warn("next5 fail", e); }

  // Top scorers (Premier por defecto; si no, LaLiga)
  try{
    let data = await getJSON("/topscorers?league=E0");
    if(!data.players?.length) data = await getJSON("/topscorers?league=SP1");
    const box = $id("topscorers");
    if(box){
      box.innerHTML = "";
      (data.players||[]).slice(0,8).forEach(p=>{
        const row = document.createElement("div");
        row.style.display="flex"; row.style.alignItems="center"; row.style.justifyContent="space-between";
        row.style.padding="10px 0"; row.style.borderBottom="1px solid var(--border)";
        row.innerHTML = `
          <div style="display:flex; align-items:center; gap:10px">
            <img src="${p.photo||""}" alt="" style="width:36px;height:36px;border-radius:50%;object-fit:cover;"/>
            <div>
              <div style="font-weight:600">${p.player_name||"—"}</div>
              <div style="color:var(--muted); font-size:12px">${p.team_name||""}</div>
            </div>
          </div>
          <div style="text-align:right">
            <div style="font-weight:700">${p.goals ?? 0}</div>
            <div style="color:var(--muted);font-size:10px;letter-spacing:.08em">GOALS</div>
          </div>`;
        box.appendChild(row);
      });
    }
  }catch(e){ warn("topscorers fail", e); }
}

/* ---------- PREDICCIÓN ---------- */
let leaguesLoaded = false;
function ensureLeagueListsBound(){
  if (leaguesLoaded) return;
  leaguesLoaded = true;

  const btnLoad = $id("btn-load-leagues");
  const selLg   = $id("sel-league");
  const selH    = $id("sel-home");
  const selA    = $id("sel-away");
  const btnPred = $id("btn-predict");

  btnLoad?.addEventListener("click", async ()=>{
    try{
      const ls = await getJSON("/leagues");
      selLg.innerHTML = "";
      (ls.leagues||[]).forEach(code => selLg.appendChild(opt(code)));
      if (selLg.value) await populateTeams(selLg.value);
    }catch(e){ alert("No se pudieron cargar ligas."); }
  });

  selLg?.addEventListener("change", async ()=>{ await populateTeams(selLg.value); });

  btnPred?.addEventListener("click", async ()=>{
    const home = selH?.value; const away = selA?.value;
    if(!home || !away || home===away){ alert("Elige dos equipos distintos."); return; }
    try{
      const r = await getJSON(`/predict?home=${encodeURIComponent(home)}&away=${encodeURIComponent(away)}`);
      renderPrediction(r);
    }catch(e){ alert("Fallo la predicción."); }
  });
}

async function populateTeams(leagueCode){
  const selH = $id("sel-home"); const selA = $id("sel-away");
  if(!leagueCode || !selH || !selA) return;
  selH.innerHTML=""; selA.innerHTML="";
  try{
    const t = await getJSON(`/teams?league=${encodeURIComponent(leagueCode)}`);
    (t.teams||[]).forEach(name=>{
      selH.appendChild(opt(name));
      selA.appendChild(opt(name));
    });
  }catch(e){ warn("teams fail", e); }
}

function renderPrediction(r){
  // Barras 1X2
  const setBar = (idBar, idLab, p) => {
    const el = $id(idBar); const lb = $id(idLab);
    if(el) el.style.height = `${asPct(p)}%`;
    if(lb) lb.textContent = `${asPct(p)}%`;
  };
  setBar("vbar-home","vlab-home", r.p_home);
  setBar("vbar-draw","vlab-draw", r.p_draw);
  setBar("vbar-away","vlab-away", r.p_away);

  // OU/BTTS barras horizontales
  const setFill = (idFill,idLab,p)=>{
    const el = $id(idFill); const lb = $id(idLab);
    if(el) el.style.width = `${asPct(p)}%`;
    if(lb) lb.textContent = `${asPct(p)}%`;
  };
  setFill("bar-ou-over","lab-ou-over", r.ou_over25);
  setFill("bar-ou-under","lab-ou-under", r.ou_under25);
  setFill("bar-btts-yes","lab-btts-yes", r.btts_yes);
  setFill("bar-btts-no","lab-btts-no", r.btts_no);

  // Píldoritas
  const setTxt = (id, v, digits=2)=>{ const el=$id(id); if(el) el.textContent = (v==null?"—":Number(v).toFixed(digits)); };
  setTxt("stat-xg-h", r.exp_goals_home);
  setTxt("stat-xg-a", r.exp_goals_away);
  setTxt("stat-xg-t", r.exp_goals_total);

  setTxt("stat-ou-o", r.ou_over25, 3);
  setTxt("stat-ou-u", r.ou_under25, 3);
  setTxt("stat-btts-y", r.btts_yes, 3);
  setTxt("stat-btts-n", r.btts_no, 3);

  setTxt("stat-c-h", r.exp_corners_home);
  setTxt("stat-c-a", r.exp_corners_away);

  setTxt("stat-y-h", r.exp_yellows_home);
  setTxt("stat-y-a", r.exp_yellows_away);

  setTxt("stat-r-h", r.exp_reds_home, 3);
  setTxt("stat-r-a", r.exp_reds_away, 3);
}

/* ---------- ESTADO DEL SISTEMA ---------- */
let statusBound = false;
function bindSystemStatusOnce(){
  if(statusBound) return;
  statusBound = true;
  const btn = $id("btn-refresh-status");
  btn?.addEventListener("click", loadStatusBox);
  loadStatusBox();
}

async function loadStatusBox(){
  try{
    const s = await getJSON("/bot_status");
    const box = $id("system-status");
    if(!box) return;
    const pill = (k,v,ok=true)=>`<span class="pill" style="margin-right:6px">${k}: <b class="${ok?'ok':'warn'}">${v}</b></span>`;
    box.innerHTML = `
      ${pill("Filas", s.data_rows ?? 0)}
      ${pill("Features", s.features_ready ? "listas":"no listas", !!s.features_ready)}
      ${pill("Poisson", s.models?.poisson ? "OK":"—", !!s.models?.poisson)}
      ${pill("ML 1X2", s.models?.ml_1x2 ? "OK":"—", !!s.models?.ml_1x2)}
      ${pill("ML OU", s.models?.ml_ou25 ? "OK":"—", !!s.models?.ml_ou25)}
      ${pill("ML BTTS", s.models?.ml_btts ? "OK":"—", !!s.models?.ml_btts)}
      <div style="margin-top:8px; color:var(--muted); font-size:12px">CSV actual: ${s.current_csv || "—"}</div>
    `;
  }catch(e){
    ($id("system-status")||{}).textContent = "No disponible.";
  }
}

/* ---------- ARRANQUE ---------- */
document.addEventListener("DOMContentLoaded", async ()=>{
  // Botones login/logout
  $id("login-submit")?.addEventListener("click", doLogin);
  $id("btn-logout")?.addEventListener("click", doLogout);

  // Navegación lateral
  document.querySelectorAll("nav .nav-btn").forEach(btn=>{
    btn.addEventListener("click", ()=>{
      const page = btn.dataset.page;
      if(!canAccess(page)){ alert("No autorizado para esta sección."); return; }
      history.replaceState(null,"",`#/${page}`);
      routeToHash();
    });
  });

  // Validar token existente con /auth/me
  try{
    const token = localStorage.getItem("sp_token");
    if (token){
      const me = await getJSON("/auth/me");  // {username, role}
      if (me?.role){
        localStorage.setItem("sp_role", me.role);
        if (me?.username) localStorage.setItem("sp_user", me.username);
      }
    }else{
      localStorage.setItem("sp_role","guest");
      localStorage.removeItem("sp_user");
    }
  }catch{
    localStorage.removeItem("sp_token");
    localStorage.setItem("sp_role","guest");
    localStorage.removeItem("sp_user");
  }

  // Pintar cabecera/menú según rol y enrrutar
  refreshUserUI();
  routeToHash();

  // Responder a cambios de hash
  window.addEventListener("hashchange", routeToHash);

  // Cargar Home al inicio
  loadHome().catch(()=>{});
});
