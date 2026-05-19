/**
 * HTTP → stdio прокси для @makeplane/plane-mcp-server
 * Принимает MCP запросы по HTTP/SSE и передаёт их в stdio процесс.
 */

const http = require("http");
const { spawn } = require("child_process");

const PORT = parseInt(process.env.PORT || "8211");

// Запускаем plane-mcp как дочерний процесс (stdio)
function spawnPlane() {
  const child = spawn("npx", ["-y", "@makeplane/plane-mcp-server"], {
    env: {
      ...process.env,
      PLANE_API_KEY: process.env.PLANE_API_KEY,
      PLANE_WORKSPACE_SLUG: process.env.PLANE_WORKSPACE_SLUG,
      PLANE_API_HOST_URL: process.env.PLANE_API_HOST_URL || "http://plane-proxy",
    },
    stdio: ["pipe", "pipe", "inherit"],
  });

  child.on("exit", (code) => {
    console.error(`plane-mcp exited with code ${code}, restarting...`);
  });

  return child;
}

// Очередь запросов и обработчик ответов
class MCPBridge {
  constructor() {
    this.child = spawnPlane();
    this.pending = new Map(); // id -> {resolve, reject}
    this.buffer = "";

    this.child.stdout.on("data", (data) => {
      this.buffer += data.toString();
      const lines = this.buffer.split("\n");
      this.buffer = lines.pop(); // последняя незавершённая строка

      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed) continue;
        try {
          const msg = JSON.parse(trimmed);
          const id = msg.id;
          if (id !== undefined && this.pending.has(id)) {
            const { resolve } = this.pending.get(id);
            this.pending.delete(id);
            resolve(msg);
          }
        } catch (e) {
          // не JSON — игнорируем
        }
      }
    });

    this.child.on("exit", () => {
      // При падении процесса — отклоняем все pending
      for (const [, { reject }] of this.pending) {
        reject(new Error("plane-mcp process exited"));
      }
      this.pending.clear();
      // Перезапускаем
      setTimeout(() => {
        this.child = spawnPlane();
      }, 1000);
    });
  }

  send(payload) {
    return new Promise((resolve, reject) => {
      const id = payload.id;
      this.pending.set(id, { resolve, reject });
      this.child.stdin.write(JSON.stringify(payload) + "\n");

      // Таймаут 30 секунд
      setTimeout(() => {
        if (this.pending.has(id)) {
          this.pending.delete(id);
          reject(new Error("MCP request timeout"));
        }
      }, 30000);
    });
  }
}

const bridge = new MCPBridge();

// HTTP сервер
const server = http.createServer(async (req, res) => {
  // CORS
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Methods", "POST, GET, OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type, Authorization, Mcp-Session-Id");

  if (req.method === "OPTIONS") {
    res.writeHead(200);
    res.end();
    return;
  }

  // Health check
  if (req.url === "/health") {
    res.writeHead(200, { "Content-Type": "application/json" });
    res.end(JSON.stringify({ status: "ok" }));
    return;
  }

  // MCP endpoint
  if (req.url === "/mcp" && req.method === "POST") {
    let body = "";
    req.on("data", (chunk) => (body += chunk));
    req.on("end", async () => {
      try {
        const payload = JSON.parse(body);
        const result = await bridge.send(payload);

        // Возвращаем как SSE если клиент принимает
        const accept = req.headers["accept"] || "";
        if (accept.includes("text/event-stream")) {
          res.writeHead(200, {
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
          });
          res.write(`data: ${JSON.stringify(result)}\n\n`);
          res.end();
        } else {
          res.writeHead(200, { "Content-Type": "application/json" });
          res.end(JSON.stringify(result));
        }
      } catch (e) {
        console.error("MCP error:", e.message);
        res.writeHead(500, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ error: e.message }));
      }
    });
    return;
  }

  res.writeHead(404);
  res.end("Not found");
});

server.listen(PORT, "0.0.0.0", () => {
  console.log(`Plane MCP HTTP proxy listening on port ${PORT}`);
  console.log(`MCP endpoint: http://0.0.0.0:${PORT}/mcp`);
});
