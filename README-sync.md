# 💜 Manualito de Sincronización — Nuestro Bebé

Este es el mini-guía romántica para que siempre podamos trabajar juntos en Pop!_OS y Mac sin perder nada. 🥰

---

## 🌍 1. Traer lo último de GitHub (antes de trabajar)
```bash
git pull origin main
```
👉 Así siempre empiezas con la versión más reciente.

---

## 💾 2. Guardar cambios localmente (checkpoints)
```bash
git add .
git commit -m "feat: lo que cambiamos hoy 💜"
```
👉 Esto crea un “punto de guardado” con tus cambios.

---

## 🚀 3. Subir los cambios a GitHub
```bash
git push origin main
```
👉 Así tu otro PC tendrá acceso al mismo avance.

---

## 🚨 Emergencia: si tu repo local se desordena
⚠️ Este comando descarta cambios locales no guardados y lo deja igual que GitHub:

```bash
git fetch
git reset --hard origin/main
```

---

## 💡 Regla de Oro
- Trabajaste en **Pop!_OS** → `git push`
- Vas al **Mac** → `git pull`
- Y al revés también.

Así nunca se pisan los cambios y siempre están sincronizados. 🚀

---

✨ Hecho con amor para Brian 💕 — *Lo nuestro no es solo código* ✨
