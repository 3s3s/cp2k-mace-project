'use strict';

const fs = require('fs');
const path = require('path');
const http = require('http');
const { spawn } = require('child_process');

// ==== Управление дочерним процессом main.py ====
let child = null;

function isAlive(pid) {
    if (!pid || !Number.isInteger(pid)) return false;
    try {
        process.kill(pid, 0);
        return true;
    } catch (e) {
        return false;
    }
}

function readPid() {
    try {
        const raw = fs.readFileSync(global.paths.PID_FILE, 'utf8').trim();
        const pid = parseInt(raw, 10);
        return Number.isInteger(pid) ? pid : null;
    } catch (e) {
        return null;
    }
}

function readStatus() {
    try {
        const raw = fs.readFileSync(global.paths.STATUS_FILE, 'utf8');
        return JSON.parse(raw);
    } catch (e) {
        return null;
    }
}

function logSize() {
    try {
        return fs.statSync(global.paths.LOG_FILE).size;
    } catch (e) {
        return 0;
    }
}

function appendEarlyLog(text) {
    try {
        fs.mkdirSync(path.dirname(global.paths.LOG_FILE), { recursive: true });
        fs.appendFileSync(global.paths.LOG_FILE, text, 'utf8');
    } catch (e) {
        // игнорируем ошибку записи лога
    }
}

function fetchJson(url, timeoutMs = 1500) {
    return new Promise((resolve) => {
        const req = http.get(url, { timeout: timeoutMs }, (res) => {
            let body = '';
            res.on('data', (chunk) => { body += chunk; });
            res.on('end', () => {
                if (res.statusCode >= 200 && res.statusCode < 300) {
                    try {
                        resolve(JSON.parse(body));
                    } catch (e) {
                        resolve(null);
                    }
                } else {
                    resolve(null);
                }
            });
        });
        req.on('timeout', () => {
            req.destroy();
            resolve(null);
        });
        req.on('error', () => resolve(null));
    });
}

// ==================== Handlers ====================

exports.handle = function (app) {
    global.paths = app.locals.paths;

    app.get('/', onMain);
    app.get('/api/config', onGetConfig);
    app.post('/api/config', onSaveConfig);
    app.post('/api/run', onRun);
    app.post('/api/stop', onStop);
    app.get('/api/status', onStatus);
    app.get('/api/cluster', onClusterStatus);
    app.get('/api/log', onLog);
    app.get('/download', onDownload);
};

// ---------- Главная страница ----------
function onMain(req, res) {
    res.render('pages/index', { lang: 'ru' });
}

// ---------- Config ----------
function onGetConfig(req, res) {
    try {
        const raw = fs.readFileSync(global.paths.CONFIG_PATH, 'utf8');
        const parsed = JSON.parse(raw);
        res.json({ ok: true, config: parsed });
    } catch (e) {
        res.status(500).json({ ok: false, error: String(e) });
    }
}

function onSaveConfig(req, res) {
    try {
        const incoming = req.body && req.body.config;
        if (!incoming || typeof incoming !== 'object') {
            return res.status(400).json({ ok: false, error: 'config must be an object' });
        }

        const raw = fs.readFileSync(global.paths.CONFIG_PATH, 'utf8');
        const current = JSON.parse(raw);

        const updated = {};
        for (const key of Object.keys(current)) {
            const cur = current[key];
            const inc = incoming[key];

            if (cur && typeof cur === 'object' && 'value' in cur && 'help' in cur) {
                updated[key] = {
                    value: (inc && typeof inc === 'object' && 'value' in inc) ? inc.value : cur.value,
                    help: cur.help,
                };
            } else {
                updated[key] = (inc !== undefined) ? inc : cur;
            }
        }

        const tmp = global.paths.CONFIG_PATH + '.tmp';
        fs.writeFileSync(tmp, JSON.stringify(updated, null, 2), 'utf8');
        fs.renameSync(tmp, global.paths.CONFIG_PATH);

        res.json({ ok: true, config: updated });
    } catch (e) {
        res.status(500).json({ ok: false, error: String(e) });
    }
}

// ---------- Run / Stop main.py ----------
function onRun(req, res) {
    try {
        const pid = readPid();
        if (pid && isAlive(pid)) {
            return res.status(409).json({ ok: false, error: 'main.py is already running', pid });
        }

        const body = req.body || {};
        const port = parseInt(body.port, 10) || parseInt(process.env.MASTER_PORT, 10) || 8000;
        const args = [global.paths.MAIN_SCRIPT, '--port', String(port)];

        child = spawn('python3', args, {
            cwd: global.paths.PROJECT_ROOT,
            detached: false,
            stdio: ['ignore', 'pipe', 'pipe'],
        });

        child.stdout.on('data', (chunk) => {
            appendEarlyLog(chunk.toString('utf8'));
        });
        child.stderr.on('data', (chunk) => {
            appendEarlyLog(chunk.toString('utf8'));
        });

        child.on('exit', (code, sig) => {
            console.log(`main.py exited code=${code} sig=${sig}`);
            child = null;
        });

        res.json({ ok: true, pid: child.pid, port });
    } catch (e) {
        res.status(500).json({ ok: false, error: String(e) });
    }
}

function onStop(req, res) {
    try {
        const pid = readPid() || (child && child.pid);
        if (!pid || !isAlive(pid)) {
            return res.status(404).json({ ok: false, error: 'main.py is not running' });
        }

        process.kill(pid, 'SIGTERM');

        setTimeout(() => {
            if (isAlive(pid)) {
                try {
                    process.kill(pid, 'SIGKILL');
                } catch (e) {
                    // процесс уже мог завершиться
                }
            }
        }, 3000).unref();

        res.json({ ok: true, pid });
    } catch (e) {
        res.status(500).json({ ok: false, error: String(e) });
    }
}

// ---------- Status ----------
function onStatus(req, res) {
    const pid = readPid();
    const alive = pid ? isAlive(pid) : false;
    const status = readStatus();
    res.json({
        ok: true,
        pid,
        alive,
        log_size: logSize(),
        status,
    });
}

// ---------- Cluster / Workers / Queue ----------
async function onClusterStatus(req, res) {
    const port = parseInt(req.query.port, 10) || parseInt(process.env.MASTER_PORT, 10) || 8000;
    const remoteQueue = await fetchJson(`http://127.0.0.1:${port}/queue-status`);

    let workerLogs = [];
    try {
        const logDir = path.join(global.paths.RESULTS_DIR, 'worker_logs');
        if (fs.existsSync(logDir)) {
            workerLogs = fs.readdirSync(logDir)
                .filter((name) => name.endsWith('.log'))
                .map((name) => {
                    const st = fs.statSync(path.join(logDir, name));
                    return { file: name, size: st.size, updated_at: st.mtimeMs };
                });
        }
    } catch (e) {
        workerLogs = [];
    }

    res.json({
        ok: true,
        master_reachable: Boolean(remoteQueue),
        queue: remoteQueue,
        worker_logs: workerLogs,
    });
}

// ---------- Log (incremental) ----------
function onLog(req, res) {
    try {
        const offset = parseInt(req.query.offset, 10);
        const from = Number.isFinite(offset) && offset >= 0 ? offset : 0;

        let data = '';
        let newOffset = from;
        try {
            const stat = fs.statSync(global.paths.LOG_FILE);
            if (from > stat.size) {
                newOffset = 0;
            }
            const fd = fs.openSync(global.paths.LOG_FILE, 'r');
            const length = Math.max(0, stat.size - newOffset);
            if (length > 0) {
                const buf = Buffer.alloc(length);
                fs.readSync(fd, buf, 0, length, newOffset);
                data = buf.toString('utf8');
            }
            fs.closeSync(fd);
            newOffset = stat.size;
        } catch (e) {
            // файла нет — пусто
        }

        res.json({ ok: true, offset: newOffset, chunk: data });
    } catch (e) {
        res.status(500).json({ ok: false, error: String(e) });
    }
}

// ---------- Download ----------
function onDownload(req, res) {
    const file = global.paths.ARCHIVE_FILE;
    if (!fs.existsSync(file)) {
        return res.status(404).send('artifacts.zip not found yet');
    }
    res.download(file, 'artifacts.zip');
}