// =============================================================================
// ecosystem.config.js - PM2 Process Manager Configuration
// =============================================================================
// Purpose: Defines all bot processes for the Linux Local Master.
//          PM2 monitors these processes and restarts them on crash.
//
// Commands:
//   pm2 start ecosystem.config.js       # Start all processes
//   pm2 stop all                        # Stop all processes
//   pm2 restart all                     # Restart all processes
//   pm2 status                          # Show process overview
//   pm2 logs                            # Live logs for all processes
//   pm2 logs btc-signal-check           # Logs for one specific process
//   pm2 monit                           # Live CPU/RAM monitor
//
// One-time setup (after installation):
//   pm2 start ecosystem.config.js
//   pm2 startup    # Generate systemd unit (run printed command as root)
//   pm2 save       # Persist current process list
// =============================================================================

const REPO = "/home/hp17/Tradingbot/Quantrivo/BITCOIN4Traders";

module.exports = {
  apps: [

    // -------------------------------------------------------------------------
    // Process 1: Signal Check (Primary Bot)
    // Checks the champion's trading signal every hour and sends Telegram alerts.
    // Mirrors the GitHub Actions "trading_bot.yml" workflow but runs locally,
    // faster and without the 15-minute time limit.
    // -------------------------------------------------------------------------
    {
      name: "btc-signal-check",
      script: "darwin_engine.py",
      interpreter: "python3",
      cwd: REPO,
      args: "--mode signal",

      // Network check: wait 30s at startup (buffer for router restart)
      wait_ready: true,
      listen_timeout: 30000,

      // Restart behavior
      autorestart: true,
      restart_delay: 5000,        // Wait 5s before restarting
      max_restarts: 20,           // Max 20 restarts (crash loop protection)
      min_uptime: "30s",          // Minimum uptime to be considered "stable"

      // Cron: every hour on the hour (same cadence as GitHub Actions)
      cron_restart: "0 * * * *",

      // Resource conservation: background execution for the evolution loop
      env: {
        NODE_ENV: "production",
        PYTHONPATH: `${REPO}/src`,
        LOCAL_MASTER: "true",     // Signals to code that we are the Local Master
        PYTHONUNBUFFERED: "1",    // Flush log output immediately
      },

      // Log file paths
      out_file: `${REPO}/logs/pm2/signal-check.log`,
      error_file: `${REPO}/logs/pm2/signal-check-error.log`,
      log_date_format: "YYYY-MM-DD HH:mm:ss",
      merge_logs: false,
    },

    // -------------------------------------------------------------------------
    // Process 2: 12h Evolution Training (Background)
    // Runs continuously in the background at lowest CPU priority.
    // nice -n 19 means "use only idle CPU" - no noticeable system impact.
    // Automatically restarts after each completed training run.
    // -------------------------------------------------------------------------
    {
      name: "btc-evolution",
      script: "auto_12h_train.py",
      interpreter: "python3",
      cwd: REPO,

      // Restart behavior: restarts automatically after each 12h run
      autorestart: true,
      restart_delay: 60000,       // 1-minute pause between runs
      max_restarts: 999,          // Practically unlimited (continuous operation)
      min_uptime: "60s",

      env: {
        PYTHONPATH: `${REPO}/src`,
        LOCAL_MASTER: "true",
        PYTHONUNBUFFERED: "1",
        // nice -n 19: only runs when other processes are idle
        NICE_LEVEL: "19",
      },

      // Log files (auto_12h_train.py also manages its own logs)
      out_file: `${REPO}/logs/pm2/evolution.log`,
      error_file: `${REPO}/logs/pm2/evolution-error.log`,
      log_date_format: "YYYY-MM-DD HH:mm:ss",

      // No cron_restart - runs to completion then restarts via autorestart
    },

    // -------------------------------------------------------------------------
    // Process 3: Champion Sync (Git Push to GitHub)
    // Synchronizes the champion to GitHub every 6 hours.
    // Keeps GitHub Actions (backup) and Colab (AI lab) up to date.
    // -------------------------------------------------------------------------
    {
      name: "btc-champion-sync",
      script: "sync_champion.sh",
      interpreter: "bash",
      cwd: REPO,

      // Cron: every 6 hours (00:00, 06:00, 12:00, 18:00 UTC)
      cron_restart: "0 */6 * * *",

      // Run once then wait for next cron trigger (not a long-running daemon)
      autorestart: false,

      env: {
        LOCAL_MASTER: "true",
      },

      out_file: `${REPO}/logs/pm2/sync.log`,
      error_file: `${REPO}/logs/pm2/sync-error.log`,
      log_date_format: "YYYY-MM-DD HH:mm:ss",
    },

  ],
};
