"use strict";
// sessionStorage is scoped to this origin (including port) and browser tab.
// Never put credentials in URLs, cookies, localStorage or console output.
const verdiTokenKey = "verdiwm-session-token";
async function verdiFetch(path, options = {}) {
  const target = new URL(path, location.origin);
  if (target.origin !== location.origin) throw new Error("仅支持本地同源请求");
  const token = sessionStorage.getItem(verdiTokenKey);
  if (!token) { location.replace("/auth"); throw new Error("请先登录"); }
  const headers = new Headers(options.headers || {});
  headers.set("Authorization", `Bearer ${token}`);
  const response = await fetch(target, { ...options, headers, credentials: "omit", redirect: "error", cache: "no-store" });
  if (response.status === 401) {
    sessionStorage.removeItem(verdiTokenKey);
    location.replace("/auth");
    throw new Error("访问令牌已失效，请重新登录");
  }
  return response;
}
const authForm = document.querySelector("#auth-form");
if (authForm) {
  authForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    const input = document.querySelector("#access-token");
    const message = document.querySelector("#message");
    const token = input.value.trim();
    input.value = "";
    try {
      const response = await fetch("/auth/session", { headers: { Authorization: `Bearer ${token}` }, credentials: "omit", redirect: "error", cache: "no-store" });
      if (!response.ok) throw new Error("访问令牌无效，请检查启动终端中的令牌文件。");
      sessionStorage.setItem(verdiTokenKey, token);
      location.replace("/");
    } catch (error) { message.textContent = error.message; }
  });
} else if (!sessionStorage.getItem(verdiTokenKey)) {
  location.replace("/auth");
}
