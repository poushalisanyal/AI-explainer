module.exports = {
  apps: [
    {
      name: "ai_concepts_explainer",
      script: "./venv/bin/uvicorn",
      args: "main:app --host 127.0.0.1 --port 8003",
      cwd: "/var/www/TV_TERM-EXPLAINER",
      interpreter: "none",
      log_date_format: "YYYY-MM-DD HH:mm:ss"
    }
  ]
};